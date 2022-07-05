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

string local_hash(float a, float b, float c){
    std::ostringstream stream_a;
    stream_a << std::fixed;
    stream_a << std::setprecision(5);
    stream_a << a;
    std::string str_a = stream_a.str();

    std::ostringstream stream_b;
    stream_b << std::fixed;
    stream_b << std::setprecision(5);
    stream_b << b;
    std::string str_b = stream_b.str();

    std::ostringstream stream_c;
    stream_c << std::fixed;
    stream_c << std::setprecision(5);
    stream_c << c;
    std::string str_c = stream_c.str();

    return str_a + '-' + str_b + '-' + str_c;
}
extern "C" void run( float * point_p, int* map_array_p,int* inverse_idx_p,int*  n_colaps_v_p,int n_point){
    int cnt = 0;
    std::map<string, int> v_dict;
    std::unordered_set<string> v_dict_key;
    for (int i_point = 0; i_point < n_point; i_point++){
        string v = local_hash(point_p[i_point * 3 + 0], point_p[i_point * 3 + 1], point_p[i_point * 3 + 2]);
        std::pair<std::unordered_set<string>::iterator, bool> r = v_dict_key.insert(v);
        if (r.second) {
            v_dict.insert(std::pair<string, int>(v, cnt));
            inverse_idx_p[cnt] = i_point;
            map_array_p[i_point] = cnt;
            cnt += 1;
        }
        else{
            map_array_p[i_point] = v_dict[v];
        }

    }
    n_colaps_v_p[0] = cnt;

}


