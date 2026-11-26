# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Use independently authored FP4 expert graphs and stable decode routing."""


def install(official, moe):
    from expert_graph import install as capture_expert
    from routing import install as group_routes

    for expert in moe.experts:
        if expert is not None:
            capture_expert(expert)
    group_routes(official, moe)
