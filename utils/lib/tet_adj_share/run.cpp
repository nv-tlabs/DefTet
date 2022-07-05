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

int min_3(int a, int b, int c){
    int r = a;
    if (b < r){
        r = b;
    }
    if (c < r){
        r = c;
    }
    return r;
}

int max_3(int a, int b, int c){
    int r = a;
    if (b > r){
        r = b;
    }
    if (c > r){
        r = c;
    }
    return r;
}

extern "C" void run( int * tet_list, int * face_edge_p, int * n_face_edge_p, int n_point, int  n_tet){

   int idx_array[12] = {0, 1, 2,
                 1, 0, 3,
                 2, 3, 0,
                 3, 2, 1};
   std::map<long long, std::vector<pair<int, int>>> face_index;
   std::unordered_set<long long> face_key;
   int idx_list[4][3];

   for(int i_tet = 0; i_tet < n_tet; i_tet++){
        for (int i=0; i < 12; i++){
            idx_list[i / 3][i % 3] = tet_list[i_tet * 4 + idx_array[i]];
        }
        int* tet = tet_list + i_tet * 4;
        for (int i_face = 0; i_face < 4; i_face++){
            int* triangle = idx_list[i_face];
            int a = idx_list[i_face][0];
            int b = idx_list[i_face][1];
            int c = idx_list[i_face][2];
            a = min_3(idx_list[i_face][0], idx_list[i_face][1], idx_list[i_face][2]);
            b = max_3(idx_list[i_face][0], idx_list[i_face][1], idx_list[i_face][2]);
            for (int i=0; i < 3; i++){
                if(a!= idx_list[i_face][i] && b!= idx_list[i_face][i]){
                    c = idx_list[i_face][i];
                }
            }

            long long num = (long long) a * (long long) n_point * (long long) n_point + (long long) b * (long long) n_point + (long long) c;
            std::pair<std::unordered_set<long long>::iterator, bool> r = face_key.insert(num);
            if(r.second == true){
                std::vector<pair<int, int>> tmp_v;
                face_index.insert(pair<long long, std::vector<pair<int, int>>>(num, tmp_v));
            }
            face_index[num].push_back(pair<int, int>(i_tet, i_face));
        }

   }
    std::map<long long, std::vector<pair<int, int>>>::iterator itr;
    int cnt = 0;
    for (itr = face_index.begin(); itr != face_index.end(); ++itr){

        std::vector<pair<int, int>> f = (*itr).second;
        if(f.size() == 2){
            face_edge_p[cnt * 6 + 0] = f[0].first;
            face_edge_p[cnt * 6 + 1] = f[1].first;
            face_edge_p[cnt * 6 + 2] = f[0].second;

            face_edge_p[cnt * 6 + 3] = f[1].first;
            face_edge_p[cnt * 6 + 4] = f[0].first;
            face_edge_p[cnt * 6 + 5] = f[1].second;
            cnt += 1;

        }
    }
    n_face_edge_p[0] = cnt;

}


