/*
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
*/
#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <map>
#include <set>
#include <bits/stdc++.h>
using namespace std;

long long get_i(int a, int b, int n){
    return (long long) a * (long long)n + (long long)b;
}
extern "C" void run( int * tet_list, int * edge_p, int * n_edge, int n_point, int  n_tet){

   std::unordered_set<long long> adj_list;

    for(int i_tet = 0; i_tet < n_tet; i_tet++){
        int * tet = tet_list + i_tet * 4;
        adj_list.insert(get_i(tet[0], tet[1], n_point));
        adj_list.insert(get_i(tet[0], tet[2], n_point));
        adj_list.insert(get_i(tet[0], tet[3], n_point));

        adj_list.insert(get_i(tet[1], tet[0], n_point));
        adj_list.insert(get_i(tet[1], tet[2], n_point));
        adj_list.insert(get_i(tet[1], tet[3], n_point));

        adj_list.insert(get_i(tet[2], tet[0], n_point));
        adj_list.insert(get_i(tet[2], tet[1], n_point));
        adj_list.insert(get_i(tet[2], tet[3], n_point));

        adj_list.insert(get_i(tet[3], tet[0], n_point));
        adj_list.insert(get_i(tet[3], tet[1], n_point));
        adj_list.insert(get_i(tet[3], tet[2], n_point));
    }

    int cnt = 0;
    std::unordered_set<long long>::iterator itr;
    for (itr = adj_list.begin(); itr != adj_list.end(); ++itr){
        long long e = *itr;
        int a = (int) (e / (long long) n_point);
        int b = (int) (e % (long long) n_point);
        edge_p[cnt * 2] = a;
        edge_p[cnt * 2 + 1] = b;
        cnt += 1;

    }

    n_edge[0] = cnt;
}


