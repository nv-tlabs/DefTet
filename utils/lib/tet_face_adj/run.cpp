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
#include <chrono>
using namespace std;

extern "C" void run( int * tet_list, int * face_edge_p, int * n_face_edge_p, int n_point, int  n_tet){


   int idx_array[12] = {0, 1, 2,
                 1, 0, 3,
                 2, 3, 0,
                 3, 2, 1};
   std::map<int, std::vector<int>> edge_idx;
   std::map<int, long long>  absolute_face_idx;
   int idx_list[4][3] = {0};
   std::unordered_set<int> edge_idx_set;
   auto start = std::chrono::high_resolution_clock::now();
   for(int i_tet = 0; i_tet < n_tet; i_tet++){
        for (int i=0; i < 12; i++){
            idx_list[i / 3][i % 3] = tet_list[i_tet * 4 + idx_array[i]];
        }
        for (int i_face = 0; i_face < 4; i_face++){
            int* triangle = idx_list[i_face];
            for (int i_edge = 0; i_edge < 3; i_edge++){
                int point_a = min(triangle[i_edge], triangle[(i_edge + 1) % 3]);
                int point_b = max(triangle[i_edge], triangle[(i_edge + 1) % 3]);
                int e = point_a * n_point + point_b;
                std::pair<std::unordered_set<int>::iterator, bool> r = edge_idx_set.insert(e);
                if(r.second == true){
                    std::vector<int> tmp_v;
                    edge_idx.insert(std::pair<int, std::vector<int>>(e, tmp_v));
                }
                edge_idx[e].push_back(i_tet * 4 + i_face);

            }
            int face_p_a = triangle[0];
            int face_p_b = triangle[0];
            int face_p_c = triangle[0];
            for (int i=0; i<3; i++){
                if (triangle[i] < face_p_a){
                    face_p_a = triangle[i];
                }
                if (triangle[i] > face_p_b){
                    face_p_b = triangle[i];
                }

            }
            for (int i=0; i<3; i++){
                if (triangle[i] != face_p_a && triangle[i] != face_p_b){
                    face_p_c = triangle[i];
                }
            }
            absolute_face_idx.insert(std::pair<int, long long>(i_tet * 4 + i_face, (long long) face_p_a * ( (long long)n_point * (long long)n_point) + (long long) face_p_b * (long long) n_point + (long long) face_p_c));
        }
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> elapsed = end - start;
//   cout << " prepare time " << elapsed.count() << '\n';
    int cnt = 0;
    std::map<int, std::vector<int>>::iterator itr;
    for (itr = edge_idx.begin(); itr != edge_idx.end(); ++itr){
        std::vector<int> f = (*itr).second;
        std::vector<int>::iterator face_a = f.begin();
        std::vector<int>::iterator face_b = f.begin();
        for (face_a = f.begin(); face_a != f.end(); ++face_a){
            for(face_b = f.begin(); face_b != f.end(); ++face_b){
                if(*face_a == *face_b) continue;
                if(absolute_face_idx[*face_a] == absolute_face_idx[*face_b]) continue;
                face_edge_p[cnt * 2] = *face_a;
                face_edge_p[cnt * 2 + 1] = *face_b;
                cnt += 1;
            }
        }
    }
    n_face_edge_p[0] = cnt;
     auto end_last = std::chrono::high_resolution_clock::now();
     elapsed = end_last - end;
//    cout << " final time " << elapsed.count() << '\n';

}


