from runners.iterative_warping.run_warp_with_averaged_flow import iterative_warp_with_averaged_flow
from runners.iterative_warping.get_averaged_depths import get_averaged_depths_main_func

def iterative_warping_runner(args):
    # 1. Get averaged flows
    iterative_warp_with_averaged_flow(args)

    # 2. Get averaged depths
    get_averaged_depths_main_func(args)
    
