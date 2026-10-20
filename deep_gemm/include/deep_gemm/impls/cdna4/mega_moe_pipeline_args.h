#pragma once
#include <deep_gemm/impls/cdna4/gemm_args.h>
#include <deep_gemm/impls/cdna4/mega_moe_args.h>
namespace deep_gemm::cdna4 {
struct MegaMoePushBlock { int expert, source_offset, dst_row, rows; };
struct MegaMoeRowBlock { int expert, m0, end, shared; };
struct MegaMoePipelineArgs {
    MegaMoeArgs moe;
    GemmArgs l1, l2, shared_l1, shared_l2;
    MegaMoeRowBlock* blocks;
    MegaMoePushBlock* push_blocks;
    int min_l1_ready, l1_workers;
    int *state, *l1_done, *l2_done, *sf_ready, *jobs, *job_ready, *combine_jobs, *combine_ready;
    uint64_t* metrics;
    uint64_t *comm_metrics, *worker_ids;
    int* l1_issued;
    uint64_t* k_ready;
    MegaMoePipelineArgs* context;
    int max_blocks, shared_tokens, input_group, dispatch_blocks, n1, n2;
};
struct MegaMoePipelineLaunch { const MegaMoePipelineArgs* context; };
}
