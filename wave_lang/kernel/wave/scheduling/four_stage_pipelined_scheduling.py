import torch.fx as fx
from .graph_utils import Edge, sort_graph_by_edge_weight
from .resources import Operation
from enum import Enum, auto
from .scheduler_utils import (
    get_scheduling_stage,
    BaseScheduler,
    is_single_mma_source,
    is_mma_node,
)
from ...compiler.kernel_codegen import filter_fx_graph
from typing import List
import logging
from ...ops.wave_ops import (
    CustomOp,
    GatherToLDS,
    MMA,
    ScaledMMA,
    Write,
    get_custom,
)
from ..utils.general_utils import is_shared_read
from collections import deque
from ..utils.classes import GemmOperationType


logger = logging.getLogger(__name__)


class FourStageStage(Enum):
    GLOBAL_LOAD = auto()
    LOCAL_STORE = auto()
    LOCAL_LOAD = auto()
    COMPUTE = auto()
    SCHEDULING_NOOP = -1

    # Helper function to check next stage follows from current.
    @staticmethod
    def is_valid_transition(
        from_stage: "FourStageStage", to_stage: "FourStageStage"
    ) -> bool:
        if from_stage == to_stage:
            return True
        return (from_stage, to_stage) in _four_stage_stage_transition_table


_four_stage_stage_transition_table = {
    (FourStageStage.GLOBAL_LOAD, FourStageStage.LOCAL_STORE),
    (FourStageStage.LOCAL_STORE, FourStageStage.LOCAL_LOAD),
    # GLOBAL_TO_SHARED combines both GLOBAL_LOAD and LOCAL_STORE
    (FourStageStage.GLOBAL_LOAD, FourStageStage.LOCAL_LOAD),
    (FourStageStage.LOCAL_LOAD, FourStageStage.COMPUTE),
    (FourStageStage.COMPUTE, FourStageStage.GLOBAL_LOAD),
}
_operation_stage_table = {
    Operation.READ_SHARED: FourStageStage.LOCAL_LOAD,
    Operation.WRITE_SHARED: FourStageStage.LOCAL_STORE,
    Operation.READ_GLOBAL: FourStageStage.GLOBAL_LOAD,
    Operation.GLOBAL_TO_SHARED: FourStageStage.GLOBAL_LOAD,
    Operation.MMA: FourStageStage.COMPUTE,
    Operation.NOOP: FourStageStage.SCHEDULING_NOOP,
    Operation.VALU: FourStageStage.COMPUTE,
    Operation.SHUFFLE: FourStageStage.COMPUTE,
    Operation.WRITE_GLOBAL: FourStageStage.COMPUTE,
}


class FourStageScheduler(BaseScheduler):
    """
    Four Stage Pipelined Scheduler

    Precondition: Only a single MMA instruction group is allowed for this scheduling approach

    Convert vanilla schedule of:
        for i = 0 to N:
            a = READ_GLOBAL i
            WRITE_SHARED a
            barrier
            b = READ_SHARED
            COMPUTE b

    let SM be shared memory, then SM[0] SM[1] are the multibuffers
    into mega pipelined schedule:
        a_0 = READ_GLOBAL 0

        WRITE_SHARED a_0 SM[0]
        a_1 = READ_GLOBAL 1

        b_0 = READ_SHARED SM[0]
        WRITE_SHARED a_1 SM[1]
        a_2 = READ_GLOBAL 2


        for i = 0 to N -3:
            COMPUTE b_i
            b_{i+1} = READ_SHARED SM[i+1 %2]
            WRITE_SHARED a_{i+2} SM[i%2]
            a_{i+3} = READ_GLOBAL i+3
            barrier


        COMPUTE b_{n-2}
        b_{n-1} = READ_SHARED SM[n-1 %2]
        WRITE_SHARED a_{n} SM[n % 2]

        COMPUTE b_{n-1}
        b_{n} = READ_SHARED SM[n %2]

        COMPUTE b_n

    """

    def four_stage_scheduling(
        self, graph: fx.Graph, edges: list[Edge]
    ) -> tuple[dict[fx.Node, int], bool]:
        """
        Classify node to different stages. Based on its stage,
        program schedules the node to a specific cycle for  each node.
        This function also checks that the sorted nodes move contiguously through
        expected stages.
        """
        sorted_nodes = sort_graph_by_edge_weight(graph.nodes, edges)
        schedule = {}
        current_stage = get_scheduling_stage(sorted_nodes[0], _operation_stage_table)

        all_mma_nodes = list()
        current_stage_idx = 0
        for node in sorted_nodes:
            if is_mma_node(node):
                all_mma_nodes.append(node)

            node_stage = get_scheduling_stage(node, _operation_stage_table)
            if node_stage in [current_stage, FourStageStage.SCHEDULING_NOOP]:
                schedule[node] = current_stage_idx
            elif FourStageStage.is_valid_transition(current_stage, node_stage):
                current_stage_idx += 1
                schedule[node] = current_stage_idx
                current_stage = node_stage
            else:
                logger.warning(
                    f"No valid transition from {current_stage} to {node_stage} for node {node}"
                )
                # Node does not move contigously through stages.
                return {}, False
        if not is_single_mma_source(all_mma_nodes):
            logger.warning(
                "Structure of kernel is different than expected, only one MMA is present"
            )
            return {}, False

        return schedule, True

    def get_closest_local_load(self, node: fx.Node):
        workqueue = deque([node])
        seen = set()
        can_extend = lambda x: isinstance(x, fx.Node) and x not in seen
        while workqueue:
            cur_node = workqueue.pop()
            if is_shared_read(get_custom(cur_node)):
                return cur_node
            child_nodes = [arg_node for arg_node in cur_node.args if can_extend(arg_node)]

            # Update ancestor and seen tracker, and workqueue with new child nodes.
            seen.update(child_nodes)
            workqueue.extend(child_nodes)
        return None

    def get_local_loads(self, mma_nodes):
        local_load_lhs = []
        local_load_rhs = []
        for mma_node in mma_nodes:
            custom = get_custom(mma_node)
            lhs = self.get_closest_local_load(custom.lhs)
            rhs = self.get_closest_local_load(custom.rhs)
            if lhs == None or rhs == None:
                return None, None
            local_load_lhs.append(lhs)
            local_load_rhs.append(rhs)
        return local_load_lhs, local_load_rhs

    def get_scale_local_loads(self, mma_nodes):
        local_load_lhs_scale = []
        local_load_rhs_scale = []
        for mma_node in mma_nodes:
            custom = get_custom(mma_node)
            lhs_scale = self.get_closest_local_load(custom.lhs_scale)
            rhs_scale = self.get_closest_local_load(custom.rhs_scale)
            if lhs_scale == None or rhs_scale == None:
                return None, None
            local_load_lhs_scale.append(lhs_scale)
            local_load_rhs_scale.append(rhs_scale)
        return local_load_lhs_scale, local_load_rhs_scale

    def get_local_writes(self, local_loads):
        local_writes = set()
        for local_load in local_loads:
            custom = get_custom(local_load)
            cur_writes = [
                w
                for w in custom.memory.users
                if isinstance(get_custom(w), Write) and w.graph == custom.graph
            ]
            local_writes.update(cur_writes)
        return list(local_writes)

    def get_lds_gathers(self, local_loads):
        lds_gathers = set()
        for local_load in local_loads:
            custom = get_custom(local_load)
            # Get direct users and users from rotated registers.
            memory_users = set([g for g in custom.memory.users])
            # Filter users for GatherToLDS
            cur_gathers = [
                g
                for g in memory_users
                if isinstance(get_custom(g), GatherToLDS) and g.graph == custom.graph
            ]
            lds_gathers.update(cur_gathers)
        return list(lds_gathers)

    def get_global_loads(self, local_writes):
        global_loads = set()
        for local_write in local_writes:
            custom = get_custom(local_write)
            global_loads.add(custom.register_)
        return list(global_loads)

    def annotate_op_with_gemm_operation_type(self, nodes, gemm_operation_type):
        for node in nodes:
            if isinstance(node, CustomOp):
                node = node.fx_node
            node.meta["prefetch_stage"] = gemm_operation_type

    def annotate_gemm_operation_type(self, graph):
        mma_nodes = filter_fx_graph(
            graph,
            lambda node: isinstance(get_custom(node), (MMA, ScaledMMA)),
        )
        # Early exit if no MMA found.
        if not mma_nodes:
            return

        mma_types = set([type(get_custom(mma_node)) for mma_node in mma_nodes])
        # Only handle single MMA per kernel and need to have same type.s
        if len(mma_types) != 1:
            return

        mma_type = mma_types.pop()
        local_load_lhs, local_load_rhs = self.get_local_loads(mma_nodes)
        # Early exit if cannot find either local loads
        if not local_load_lhs or not local_load_rhs:
            return
        global_to_shared_lhs = self.get_lds_gathers(local_load_lhs)
        global_to_shared_rhs = self.get_lds_gathers(local_load_rhs)
        local_write_lhs = self.get_local_writes(local_load_lhs)
        local_write_rhs = self.get_local_writes(local_load_rhs)
        global_load_lhs = self.get_global_loads(local_write_lhs)
        global_load_rhs = self.get_global_loads(local_write_rhs)

        self.annotate_op_with_gemm_operation_type(local_load_lhs, GemmOperationType.LOCAL_LOAD_LHS)
        self.annotate_op_with_gemm_operation_type(local_load_rhs, GemmOperationType.LOCAL_LOAD_RHS)
        self.annotate_op_with_gemm_operation_type(global_to_shared_lhs, GemmOperationType.GLOBAL_LOAD_TO_LDS_LHS)
        self.annotate_op_with_gemm_operation_type(global_to_shared_rhs, GemmOperationType.GLOBAL_LOAD_TO_LDS_RHS)
        self.annotate_op_with_gemm_operation_type(local_write_lhs, GemmOperationType.LOCAL_WRITE_LHS)
        self.annotate_op_with_gemm_operation_type(local_write_rhs, GemmOperationType.LOCAL_WRITE_RHS)
        self.annotate_op_with_gemm_operation_type(global_load_lhs, GemmOperationType.GLOBAL_LOAD_LHS)
        self.annotate_op_with_gemm_operation_type(global_load_rhs, GemmOperationType.GLOBAL_LOAD_RHS)

        if mma_type == ScaledMMA:
            local_load_lhs_scale, local_load_rhs_scale = self.get_scale_local_loads(
                mma_nodes
            )
            global_to_shared_lhs_scale = self.get_lds_gathers(local_load_lhs_scale)
            global_to_shared_rhs_scale = self.get_lds_gathers(local_load_rhs_scale)
            local_write_lhs_scale = self.get_local_writes(local_load_lhs_scale)
            local_write_rhs_scale = self.get_local_writes(local_load_rhs_scale)
            global_load_lhs_scale = self.get_global_loads(local_write_lhs_scale)
            global_load_rhs_scale = self.get_global_loads(local_write_rhs_scale)

            self.annotate_op_with_gemm_operation_type(local_load_lhs_scale, GemmOperationType.LOCAL_LOAD_LHS_SCALE)
            self.annotate_op_with_gemm_operation_type(local_load_rhs_scale, GemmOperationType.LOCAL_LOAD_RHS_SCALE)
            self.annotate_op_with_gemm_operation_type(global_to_shared_lhs_scale, GemmOperationType.GLOBAL_LOAD_TO_LDS_LHS_SCALE)
            self.annotate_op_with_gemm_operation_type(global_to_shared_rhs_scale, GemmOperationType.GLOBAL_LOAD_TO_LDS_RHS_SCALE)
            self.annotate_op_with_gemm_operation_type(local_write_lhs_scale, GemmOperationType.LOCAL_WRITE_LHS_SCALE)
            self.annotate_op_with_gemm_operation_type(local_write_rhs_scale, GemmOperationType.LOCAL_WRITE_RHS_SCALE)
            self.annotate_op_with_gemm_operation_type(global_load_lhs_scale, GemmOperationType.GLOBAL_LOAD_LHS_SCALE)
            self.annotate_op_with_gemm_operation_type(global_load_rhs_scale, GemmOperationType.GLOBAL_LOAD_RHS_SCALE)
        self.annotate_op_with_gemm_operation_type(mma_nodes, GemmOperationType.MMA)


    def schedule_graph(self) -> tuple[dict[fx.Node, int], bool]:
        """
        1. Identify which nodes are part of the global_read/local_write/local_read/compute phase
        2. Set nodes to clock (0,1,2,3) based on phase.
        3. Set initiation interval to 1.
        """
        self.annotate_gemm_operation_type(self.graph)
        self.schedule, success = self.four_stage_scheduling(self.graph, self.edges)
        self._initiation_interval = 1
        return self.schedule, success
