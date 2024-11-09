#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <typeinfo>
#include <fstream>
#include <hdf5.h>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <set>
#include <map>
#include <cmath>
#include <iomanip>
#include <omp.h>
#define WRITE_ATTRIBUTE(attr_name, attr_value, attr_type)                                           \
  dataspace_id = H5Screate(H5S_SCALAR);                                                             \
  attribute_id = H5Acreate(group_id, attr_name, attr_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); \
  status       = H5Awrite(attribute_id, attr_type, &(attr_value));                                  \
  status       = H5Aclose(attribute_id);                                                            \
  status       = H5Sclose(dataspace_id);
const int N = 512;
constexpr int N3 = N * N * N;

int thread, n_threads;
#pragma omp threadprivate(thread)

using namespace std;
struct Coordinate {
    int x, y, z;

    // Equality operator
    bool operator==(const Coordinate& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};
struct CoordinateHash {
    std::size_t operator()(const Coordinate& p) const {
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1) ^ (std::hash<int>()(p.z) << 2);
    }
};
struct DecreasingOrder {
    bool operator()(const float& lhs, const float& rhs) const {
        return lhs > rhs;
    }
};
struct bubble_group {
    // 2D vector where each vector represents the coordinates of that bubble expanded into
    // Used for store the points the bubble only expanded to
    vector<vector<int>>expansion_points_coords;
    // Vector of Unordered maps where each unordered map corresponds to its group. They keys are coordinates that neighbor the 
    // bubble groups coordinates, and the values are the corresponding values.
    // Used to easily check uniquness of neighboring points when merging
    vector<unordered_map<Coordinate, float, CoordinateHash>> neighboring_coords_and_points;
    // Vector of ordered multimaps where the keys are the values, and the values are the corresponding coordinate
    // Used to queue the next points of expansion
    vector<multimap<float, Coordinate, DecreasingOrder>> ordered_neighboring_values;
    vector<float> expansion_queue_pointer;
    // 2D vector where each vector represents the values of that bubble coordinates
    // Used to store all of the values the bubble encompases
    vector<vector<float>>values;
    // 1D vector where each index corresponds to that bubble groups size
    vector<int> size;
    // 1D vector that decreases in size to represent bubble that haven't merged into another one
    vector<int> non_merged_groups;
    // Int value to hold the total number of initial groups
    int total_group_num;
};
struct data_to_save {
    // For each 1D vector, the index corresponds to that group. So, index 5 is group 5.

    // Vector that stores what group each group merged with. For example, if group 3 merges with group 10,
    // the 3rd index of merged_with will have a value of 10.
    vector<int> merged_with;
    // Stores the z-value that group merged with another group.
    vector<float> z_merge;
    // Stores the z-value that group formed.
    vector<float> z_form;
    // Stores the number of cells the parent had when that group merged.
    vector<int> parent_cells_merged;
    // Stores the number of cells that group had when it merged.
    vector<int> cells_merged;
    // Stores the number of cells that group expanded into. Only includes expansion points, not merging points
    vector<int> counts;
    // Offsets to get the coordinates of each bubble
    vector<int> offsets;
    // Stores an array of coordinates for each bubble, offsets used to get the coordinates of each bubble.
    // 1D array where each grouping of three is one coordinate. So, index 0, 1, and 2 are all one coordinate
    vector<int> bubble_cells;
    // Stores the x,y, and z components of the r_com
    vector<float> dr_com;
    // Stores the intertia tensor of the bubble groups
    vector<float> I_com;
    // Stores the number of ionized cells for that range of z-values (ranges in HII_Z_Values)
    // Cumsummed at end to get the ionization history
    vector<int> HII_Z_count;
    // Tracks the effective radii of each point when it gets expanded to
    vector<int> eff_volume;
    // Z that effective volume gets added
    vector<float> eff_volume_z;
    // The ranges for calculating number of ionized cells
    vector<float> HII_Z_Values{ 17., 16.82608696, 16.65217391, 16.47826087, 16.30434783,
        16.13043478, 15.95652174, 15.7826087, 15.60869565, 15.43478261,
        15.26086957, 15.08695652, 14.91304348, 14.73913043, 14.56521739,
        14.39130435, 14.2173913, 14.04347826, 13.86956522, 13.69565217,
        13.52173913, 13.34782609, 13.17391304, 13., 12.82608696,
        12.65217391, 12.47826087, 12.30434783, 12.13043478, 11.95652174,
        11.7826087, 11.60869565, 11.43478261, 11.26086957, 11.08695652,
        10.91304348, 10.73913043, 10.56521739, 10.39130435, 10.2173913,
        10.04347826, 9.86956522, 9.69565217, 9.52173913, 9.34782609,
        9.17391304, 9., 8.82608696, 8.65217391, 8.47826087,
        8.30434783, 8.13043478, 7.95652174, 7.7826087, 7.60869565,
        7.43478261, 7.26086957, 7.08695652, 6.91304348, 6.73913043,
        6.56521739, 6.39130435, 6.2173913, 6.04347826, 5.86956522,
        5.69565217, 5.52173913, 5.34782609, 5.17391304, 5. };
    // Index to store which Z-value boundary for ionized cells are at
    int HII_index = 0;
    // At each coordinate, shows which bubble group into it.
    int cell_to_bubble[N][N][N];
    data_to_save(int n) {
        eff_volume.reserve(N3);
        eff_volume_z.reserve(N3);
        merged_with.resize(n);
        fill(merged_with.begin(), merged_with.end(), -1);
        z_merge.resize(n);
        z_form.resize(n);
        parent_cells_merged.resize(n);
        cells_merged.resize(n);
        counts.resize(n);
        fill(counts.begin(), counts.end(), 1);
        offsets.resize(n + 1);
        HII_Z_count.resize(70);
        fill(HII_Z_count.begin(), HII_Z_count.end(), 0);
        int* cell_to_bubble_ptr = &cell_to_bubble[0][0][0];
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N3; i++)
            cell_to_bubble_ptr[i] = -1;
    }
};

// This function takes in a 3D point r, and finds all of the adjacent coordinates. 
// If corners is true, then it also computes the adjacent corners.
//The 3D space is closed, so for a sized N space, at the coordinate value N, it goes to the 0 index.
std::vector<int> get_surrounding_coordinates(vector<int> r, bool corners) {
    vector<int> coordinate;
    coordinate.reserve(3 * 26);
    int x_increase;
    int y_increase;
    int z_increase;
    // Get all surrounding points, including corners, and skipping self.
    if (corners) {
        for (int dx = -1; dx < 2; dx++) {
            for (int dy = -1; dy < 2; dy++) {
                for (int dz = -1; dz < 2; dz++) {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;
                    x_increase = r[0] + dx;
                    y_increase = r[1] + dy;
                    z_increase = r[2] + dz;
                    if (x_increase < 0) {
                        x_increase = N + x_increase;
                    }
                    if (y_increase < 0) {
                        y_increase = N + y_increase;
                    }
                    if (z_increase < 0) {
                        z_increase = N + z_increase;
                    }
                    // Do % N because grid is closed
                    coordinate.push_back(x_increase % N);
                    coordinate.push_back(y_increase % N);
                    coordinate.push_back(z_increase % N);

                }
            }

        }
    }
    return coordinate;
}
bool check_if_bubble(float zreion[][N][N], vector<int>& points_equal, vector<int>& visited) {
    Coordinate coord;
    vector<int> r = { points_equal[0] ,points_equal[1],points_equal[2] };
    float value = zreion[r[0]][r[1]][r[2]];
    vector<int> coords_next = get_surrounding_coordinates(r, true);
    for (int i = 0; i < coords_next.size(); i += 3) {
        // if a neighboring point is greater than, then don't include it in the bubble groups
        bool already_counted = false;
        for (int j = 0; j < visited.size(); j += 3) {
            if (coords_next[i] == visited[j] && coords_next[i + 1] == visited[j + 1] && coords_next[i + 2] == visited[j + 2]) {
                already_counted=true;
                break;
            }
        }
        if (already_counted) { continue; }
        
        if (zreion[coords_next[i]][coords_next[i + 1]][coords_next[i + 2]] > value) {
            return false;
        }
        if (zreion[coords_next[i]][coords_next[i + 1]][coords_next[i + 2]] == value) {
            points_equal.push_back(i);
            points_equal.push_back(i+1);
            points_equal.push_back(i+2);
        }
    }
    visited.push_back(r[0]);
    visited.push_back(r[1]);
    visited.push_back(r[2]);

    auto first_coord = points_equal.begin();
    points_equal.erase(first_coord);
    first_coord = points_equal.begin();
    points_equal.erase(first_coord);
    first_coord = points_equal.begin();
    points_equal.erase(first_coord);

    return true;
}
// This function finds and returns the starting bubble groups.
// It does so by finding the coordinates in the dataset that are local maxima,
// then returns those as the bubble groups
bubble_group get_bubble_groups(float zreion[][N][N]) {
    bubble_group bubble_groups;
    int current_group_num = 0;
    bool is_greater = false;

    // Loop through entire dataset to check maxima
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                float value = zreion[i][j][k];
                vector<int> r{ i,j,k };
                vector<int> coords_next = get_surrounding_coordinates(r, true); // get neighboring coordinates
                for (int l = 0; l < coords_next.size(); l += 3) {
                    // if a neighboring point is greater than, then don't include it in the bubble groups
                    if (zreion[coords_next[l]][coords_next[l + 1]][coords_next[l + 2]] >= value) {
                        // If is is equal, check if still local maxima. 
                        if (zreion[coords_next[l]][coords_next[l + 1]][coords_next[l + 2]] == value) {
                            vector<int> points_equal = { i,j,k };
                            vector<int> visited = {};
                            bool keep_going;
                            keep_going = check_if_bubble(zreion, points_equal, visited);
                            while (keep_going && !points_equal.empty()) {
                                keep_going = check_if_bubble(zreion, points_equal, visited);
                            }

                            if (keep_going) {
                                vector<int> new_coords;
                                vector<float> new_values;
                                for (int q = 0; q < visited.size(); q += 3) {
                                    vector<int> curr_coord = { visited[q],visited[q + 1],visited[q + 2] };
                                    new_coords.insert(new_coords.end(), curr_coord.begin(), curr_coord.end());
                                    new_values.push_back(value);
                                }
                                bubble_groups.expansion_points_coords.push_back(new_coords);
                                bubble_groups.values.push_back(new_values);
                                bubble_groups.size.push_back(new_values.size());
                                bubble_groups.non_merged_groups.push_back(current_group_num);
                                unordered_map<Coordinate, float, CoordinateHash> pointValueMap;
                                bubble_groups.neighboring_coords_and_points.push_back(pointValueMap);
                                multimap<float, Coordinate, DecreasingOrder> orderedValues;
                                bubble_groups.ordered_neighboring_values.push_back(orderedValues);
                                bubble_groups.expansion_queue_pointer.push_back(0.0f);
                                current_group_num += 1;
                            }
                        }
                        is_greater = true;
                        break;
                    }
                }
                if (is_greater) {
                    is_greater = false;
                    continue;
                }
                vector<float> value_vec = { value };
                bubble_groups.expansion_points_coords.push_back(r);
                bubble_groups.values.push_back(value_vec);
                bubble_groups.size.push_back(1);
                bubble_groups.non_merged_groups.push_back(current_group_num);
                unordered_map<Coordinate, float, CoordinateHash> pointValueMap;
                bubble_groups.neighboring_coords_and_points.push_back(pointValueMap);
                multimap<float, Coordinate, DecreasingOrder> orderedValues;
                bubble_groups.ordered_neighboring_values.push_back(orderedValues);
                bubble_groups.expansion_queue_pointer.push_back(0.0f);
                current_group_num += 1;
            }
        }
    }
    bubble_groups.total_group_num = current_group_num;
    return bubble_groups;
}
// Gets the neighboring points for a certain coordinate in a group. If not already a neighbor, add it.
void find_neighbors(vector<int> r, int group, float zreion[][N][N], bubble_group& bubble_groups, data_to_save& saved_data) {
    vector<int> coords = get_surrounding_coordinates(r, true);
    vector<Coordinate> new_points;

    // Check surrounding coordinates to see if they are already accounted for
    for (int i = 0; i < coords.size(); i += 3) { //Hash table Unordered set
        Coordinate point = { coords[i] ,coords[i + 1],coords[i + 2] };

        // Check neighbors if it is not already a point in the bubble_group
        if (saved_data.cell_to_bubble[coords[i]][coords[i + 1]][coords[i + 2]] == -1) {
            // Check the unordered map to see if it is already there
            bool is_not_in = bubble_groups.neighboring_coords_and_points[group].find(point) == bubble_groups.neighboring_coords_and_points[group].end();
            if (is_not_in) {
                bubble_groups.neighboring_coords_and_points[group][point] = zreion[coords[i]][coords[i + 1]][coords[i + 2]];
                bubble_groups.ordered_neighboring_values[group].emplace(zreion[coords[i]][coords[i + 1]][coords[i + 2]], point);
            }
        }
    }
}

// Get the expansion queue for a certain group
void find_expansion_queue(int group, bubble_group& bubble_groups, multimap<float, int, DecreasingOrder>& expansion_queue) {
    // Since ordered_neighboring_values is a map decreasing in value, just get the first element
    auto it = bubble_groups.ordered_neighboring_values[group].begin();
    expansion_queue.emplace(it->first, group);
    bubble_groups.expansion_queue_pointer[group] = it->first;
}

// Merges all groups that need to be merged
void merge_groups(vector<int> groups_to_merge, vector<int> points_to_merge, float current_z, float zreion[][N][N], bubble_group& bubble_groups, data_to_save& saved_data, multimap<float, int, DecreasingOrder>& expansion_queue, int epoch) {
    // If the same group merges several times, this gives us how many times it merged so that we can check
    // for repeated merging points (i.e. if three groups merge into the same point)
    vector<int> offset_points;
    offset_points.resize(bubble_groups.total_group_num);

    // Loop through all group pairs to merge together
    for (int i = 0; i < groups_to_merge.size(); i += 2) {
        int larger_group;
        int smaller_group;
        // Get the group that is larger between the two groups merging
        auto start = std::chrono::high_resolution_clock::now();
        if (bubble_groups.size[groups_to_merge[i]] >= bubble_groups.size[groups_to_merge[i + 1]]) {
            larger_group = groups_to_merge[i];
            smaller_group = groups_to_merge[i + 1];
        }
        else {
            larger_group = groups_to_merge[i + 1];
            smaller_group = groups_to_merge[i];
        }

        // Update the necessary saved_data characteristics
        saved_data.merged_with[smaller_group] = larger_group;
        saved_data.z_merge[smaller_group] = current_z;
        saved_data.cells_merged[smaller_group] = bubble_groups.size[smaller_group];
        saved_data.parent_cells_merged[smaller_group] = bubble_groups.size[larger_group];


        // Get the coord position and add it to expansion points and update cell_to_bubble
        int coord_pos = i / 2 * 3;
        saved_data.cell_to_bubble[points_to_merge[coord_pos]][points_to_merge[coord_pos + 1]][points_to_merge[coord_pos + 2]] = larger_group;

        // Check if point already added to the group (i.e. if more than one group is expanding into the same point   
        auto pos_to_start = bubble_groups.expansion_points_coords[larger_group].end() - 3 * offset_points[larger_group];
        bool not_in = true;
        for (int k = 0; k < 3 * offset_points[larger_group]; k += 3) {
            if (*(pos_to_start + k) == points_to_merge[coord_pos] and *(pos_to_start + k + 1) == points_to_merge[coord_pos + 1] and *(pos_to_start + k + 2) == points_to_merge[coord_pos + 2]) {
                not_in = false;
                break;
            }

        }
        pos_to_start = bubble_groups.expansion_points_coords[smaller_group].end() - 3 * offset_points[smaller_group];
        for (int k = 0; k < 3 * offset_points[smaller_group]; k += 3) {
            if (*(pos_to_start + k) == points_to_merge[coord_pos] and *(pos_to_start + k + 1) == points_to_merge[coord_pos + 1] and *(pos_to_start + k + 2) == points_to_merge[coord_pos + 2]) {
                not_in = false;
                break;
            }

        }
        // Update data if it still needs to be added.
        if (not_in) {
            for (int j = coord_pos; j < coord_pos + 3; j++)
                bubble_groups.expansion_points_coords[larger_group].push_back(points_to_merge[j]);
            saved_data.counts[larger_group] += 1;
            bubble_groups.size[larger_group] += 1;
            offset_points[larger_group] += 1;
        }

        // Merge the groups, increasing the size accordingly and combining the coords.
        // Need to check if the smaller group hasn't already merged because it is possible to merge with
        // the large group many times.
        auto group_to_remove = find(bubble_groups.non_merged_groups.begin(), bubble_groups.non_merged_groups.end(), smaller_group);
        if (group_to_remove != bubble_groups.non_merged_groups.end()) {
            bubble_groups.size[larger_group] += bubble_groups.size[smaller_group];
            
            vector<Coordinate> neighbors_coords_to_add;
            vector<float> neighbors_values_to_add;
            for (const auto& entry : bubble_groups.neighboring_coords_and_points[smaller_group]) {
                if (bubble_groups.neighboring_coords_and_points[larger_group].find(entry.first) == bubble_groups.neighboring_coords_and_points[larger_group].end()) {
                    neighbors_coords_to_add.push_back(entry.first);
                    neighbors_values_to_add.push_back(entry.second);
                }
            }
            if (!bubble_groups.neighboring_coords_and_points[smaller_group].empty() && !bubble_groups.ordered_neighboring_values[smaller_group].empty()) {
                bubble_groups.neighboring_coords_and_points[smaller_group].clear();
                bubble_groups.ordered_neighboring_values[smaller_group].clear();
            }
            for (int j = 0; j < neighbors_coords_to_add.size(); j++) {
                bubble_groups.neighboring_coords_and_points[larger_group][neighbors_coords_to_add[j]] = neighbors_values_to_add[j];
                bubble_groups.ordered_neighboring_values[larger_group].insert({ neighbors_values_to_add[j] , neighbors_coords_to_add[j] });
            }
        }

        bool same_point_ahead = false;
        for (int j = coord_pos + 3; j < points_to_merge.size(); j+=3) {
            if (points_to_merge[j] == points_to_merge[coord_pos] && points_to_merge[j + 1] == points_to_merge[coord_pos + 1] && points_to_merge[j + 2] == points_to_merge[coord_pos + 2]) {
                same_point_ahead = true;
                break;
            }

        }
        if (!same_point_ahead) {
            saved_data.eff_volume.push_back(bubble_groups.size[larger_group]);
            saved_data.eff_volume_z.push_back(current_z);
        }
        

        // Remove the merged group from the list of expanding groups, get the neighbor of the new point, and update expansion_queue.
        // A larger group might merge with the smaller group many times, so check if the smaller group hasn't already merged. 
        if (group_to_remove != bubble_groups.non_merged_groups.end()) {
            bubble_groups.non_merged_groups.erase(group_to_remove);
        }

        // Only need to check the neighbor for the new coordinate once.
        if (not_in) {
            vector<int> r{ points_to_merge[coord_pos],points_to_merge[coord_pos + 1],points_to_merge[coord_pos + 2] };
            find_neighbors(r, larger_group, zreion, bubble_groups, saved_data);
        }
        if (bubble_groups.expansion_queue_pointer[larger_group] == 0.0f) {
            if (!bubble_groups.ordered_neighboring_values[larger_group].empty())
                find_expansion_queue(larger_group, bubble_groups, expansion_queue);
        }
        else {
            auto new_expansion = bubble_groups.ordered_neighboring_values[larger_group].begin();
            if (new_expansion->first > bubble_groups.expansion_queue_pointer[larger_group]) {
                pair<float, int> current_pair = { bubble_groups.expansion_queue_pointer[larger_group] ,larger_group };
                auto current_expansion = find(expansion_queue.begin(), expansion_queue.end(), current_pair);
                expansion_queue.erase(current_expansion);
                if (!bubble_groups.ordered_neighboring_values[larger_group].empty())
                    find_expansion_queue(larger_group, bubble_groups, expansion_queue);
            }
        }
        if (bubble_groups.expansion_queue_pointer[smaller_group] != 0.0f) {
            pair<float, int> current_pair = { bubble_groups.expansion_queue_pointer[smaller_group] ,smaller_group };
            auto current_expansion = find(expansion_queue.begin(), expansion_queue.end(), current_pair);
            expansion_queue.erase(current_expansion);
            bubble_groups.expansion_queue_pointer[smaller_group] = 0.0f;
        }
    }
}
// Given a merging list, update it so we do not get any error. 
// I.e., if group 1 merges with 2 then 2 merges with 3. And 1 is greater than 2,
// 2 will no longer exist for the 2 and 3 merger. Thus we replace the 2 with 1.
void update_merge_list(vector<int>& groups_to_merge, bubble_group& bubble_groups) {
    int size = groups_to_merge.size();
    // loop through all group pairs in the merging groups
    for (int i = 0; i < size; i += 2) {
        int larger_group;
        int smaller_group;
        // Get the group that is larger between the two groups merging
        if (bubble_groups.size[groups_to_merge[i]] >= bubble_groups.size[groups_to_merge[i + 1]]) {
            larger_group = groups_to_merge[i];
            smaller_group = groups_to_merge[i + 1];
        }
        else {
            larger_group = groups_to_merge[i + 1];
            smaller_group = groups_to_merge[i];
        }

        // Loop through all other group pairs
        for (int j = i + 2; j < size; j += 2) {
            // If the larger_group is already there, then skip it (as nothing to replace)
            if (groups_to_merge[j] == larger_group || groups_to_merge[j + 1] == larger_group) {
                continue;
            }
            //If either pair equals the smaller group, replace it with the larger group.
            if (groups_to_merge[j] == smaller_group) {
                groups_to_merge[j] = larger_group;
            }
            else if (groups_to_merge[j + 1] == smaller_group) {
                groups_to_merge[j + 1] = larger_group;
            }

        }
    }
}

// Expands the bubble_groups for the biggest neighbor. If groups expand into the same space, it merges them.
void expand(float& z, float zreion[][N][N], bubble_group& bubble_groups, data_to_save& saved_data, multimap<float, int, DecreasingOrder>& expansion_queue, int epoch) {
    float max_z = 0;

    vector<int> expanded_groups = {};
    vector<int> non_merging_coords = {};
    vector<int> non_merging_groups = {};
    vector<int> merging_groups = {};
    vector<int> merging_coords = {};

    // Gets the biggest neighbor from the expansion_queue and the corresponding groups.
    auto highest_expansion = expansion_queue.begin();
    max_z = highest_expansion->first;
    float curr_val = highest_expansion->first;
    while (curr_val == max_z) {
        expanded_groups.push_back(highest_expansion->second);
        ++highest_expansion;
        if (highest_expansion == expansion_queue.end())
            break;
        curr_val = highest_expansion->first;
    }
    expansion_queue.erase(max_z);
    z = max_z;
    // Loops through the groups to expand and expands/merges accordingly
    for (int i : expanded_groups) {
        bubble_groups.expansion_queue_pointer[i] = 0.0f;

        // Get where the expansion point is in the group
        // Since ordered_neighboring_values is an ordered map, just take the first ones
        vector<Coordinate> position;
        auto it = bubble_groups.ordered_neighboring_values[i].begin();
        while (it->first == max_z) {
            position.push_back(it->second);
            ++it;
            if (it == bubble_groups.ordered_neighboring_values[i].end())
                break;
        }

        for (int j = 0; j < position.size(); j++) {
            bubble_groups.neighboring_coords_and_points[i].erase(position[j]);
            bubble_groups.ordered_neighboring_values[i].erase(max_z);

            // Check if the point has already been expanded to. It not, add it to itself
            // If yes, begin the process to merging
            if (saved_data.cell_to_bubble[position[j].x][position[j].y][position[j].z] == -1) {
                // Add data to the non_merging_groups and coords
                non_merging_groups.push_back(i);

                non_merging_coords.push_back(position[j].x);
                non_merging_coords.push_back(position[j].y);
                non_merging_coords.push_back(position[j].z);
                saved_data.cell_to_bubble[position[j].x][position[j].y][position[j].z] = i;
            }
            else {
                // Now starting the merging process. Since, for the the other group, the position was -1, need to remove this from non_merging_groups. 
                merging_groups.push_back(saved_data.cell_to_bubble[position[j].x][position[j].y][position[j].z]);
                merging_groups.push_back(i);
                vector<int> pos_to_remove;
                for (int k = 0; k < non_merging_groups.size(); k++) {
                    if (non_merging_groups[k] == saved_data.cell_to_bubble[position[j].x][position[j].y][position[j].z]) {
                        int pos = k * 3;
                        if (non_merging_coords[pos] == position[j].x && non_merging_coords[pos + 1] == position[j].y && non_merging_coords[pos + 2] == position[j].z) {
                            pos_to_remove.push_back(k);
                            break;
                        }
                    }
                }
                // Now, remove the positions found above from the non_merging vectors.
                for (int k = 0; k < pos_to_remove.size(); k++) {
                    int updated = pos_to_remove[k] - k;
                    non_merging_groups.erase(non_merging_groups.begin() + updated);
                    for (int l = 0; l < 3; l++) {
                        auto it = non_merging_coords.begin() + updated * 3;
                        non_merging_coords.erase(it);
                    }
                }
                merging_coords.push_back(position[j].x);
                merging_coords.push_back(position[j].y);
                merging_coords.push_back(position[j].z);
            }
        }
    }
    // Update z_value boundary if moved past
    int index = 0;
    while (max_z <= saved_data.HII_Z_Values[index])
        index += 1;

    // Loop through merging points to see how many unique merging points are being added
    int mergers_to_add = merging_groups.size() / 2;
    vector<int> already_counted;
    if (merging_coords.size() > 3) {
        for (int i = 0; i < merging_coords.size(); i += 3)
            for (int j = i + 3; j < merging_coords.size(); j += 3) {
                if (merging_coords[i] == merging_coords[j] && merging_coords[i + 1] == merging_coords[j + 1] && merging_coords[i + 2] == merging_coords[j + 2]) {
                    bool not_in = true;
                    for (int k : already_counted)
                        if (j == k) {
                            not_in = false;
                            break;
                        }
                    if (not_in) {
                        mergers_to_add -= 1;
                        already_counted.push_back(j);
                    }

                }
            }
    }
    // Add all unique points that are expanding
    saved_data.HII_Z_count[index] += non_merging_groups.size() + mergers_to_add;

    // For the groups that do not merge, loop through the coords and do the necessary processes.
    for (int i = 0; i < non_merging_coords.size(); i += 3) {
        vector<int> coord{ non_merging_coords[i],non_merging_coords[i + 1],non_merging_coords[i + 2] };
        int group = non_merging_groups[i / 3];

        // Save the necessary saved_data for this group
        saved_data.cell_to_bubble[coord[0]][coord[1]][coord[2]] = group;
        saved_data.counts[group] += 1;

        // Update the group for this new coordinate
        bubble_groups.expansion_points_coords[group].push_back(coord[0]);
        bubble_groups.expansion_points_coords[group].push_back(coord[1]);
        bubble_groups.expansion_points_coords[group].push_back(coord[2]);
        bubble_groups.size[group] += 1;
        saved_data.eff_volume.push_back(bubble_groups.size[group]);
        saved_data.eff_volume_z.push_back(max_z);

        find_neighbors(coord, group, zreion, bubble_groups, saved_data);
        // Check if already added a new expansion in the queue. If so, check if new largest is greater than old.
        // This happens whenever a group has more than one expansion at the same time.
        if (bubble_groups.expansion_queue_pointer[group] == 0.0f) {
            if (!bubble_groups.ordered_neighboring_values[group].empty())
                find_expansion_queue(group, bubble_groups, expansion_queue);
        }
        else {
            auto new_expansion = bubble_groups.ordered_neighboring_values[group].begin();
            if (new_expansion->first > bubble_groups.expansion_queue_pointer[group]) {
                pair<float, int> current_pair = { bubble_groups.expansion_queue_pointer[group] ,group };
                auto current_expansion = find(expansion_queue.begin(), expansion_queue.end(), current_pair);
                expansion_queue.erase(current_expansion);
                if (!bubble_groups.ordered_neighboring_values[group].empty())
                    find_expansion_queue(group, bubble_groups, expansion_queue);
            }
        }
        
    }
    // If we have merged, then merge the groups
    if (merging_groups.size() > 0) {
        update_merge_list(merging_groups, bubble_groups);
        merge_groups(merging_groups, merging_coords, max_z, zreion, bubble_groups, saved_data, expansion_queue, epoch);
    }
}
// Write 1D float data to the hdf5 file
static void write_1d(hid_t file_id, hsize_t n_cols, float* data, const char* dataset_name)
{
    // Identifier
    hid_t dataspace_id, dataset_id;
    hsize_t dims1d[1] = { n_cols };

    dataspace_id = H5Screate_simple(1, dims1d, NULL);
    dataset_id = H5Dcreate(file_id, dataset_name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Open dataset and get dataspace
    hid_t dataset = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dataset);

    // File hyperslab
    hsize_t file_offset[1] = { 0 };
    hsize_t file_count[1] = { n_cols };

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, file_offset, NULL, file_count, NULL);

    // Memory hyperslab
    hsize_t mem_offset[1] = { 0 };
    hsize_t mem_count[1] = { n_cols };

    hid_t memspace = H5Screate_simple(1, mem_count, NULL);
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, NULL, mem_count, NULL);

    // Write
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, data);

    // Close handles
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dataset);
}
// Write 1D int data to the hdf5 file
static void write_1d(hid_t file_id, hsize_t n_cols, int* data, const char* dataset_name)
{
    // Identifier
    hid_t dataspace_id, dataset_id;
    hsize_t dims1d[1] = { n_cols };

    dataspace_id = H5Screate_simple(1, dims1d, NULL);
    dataset_id = H5Dcreate(file_id, dataset_name, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Open dataset and get dataspace
    hid_t dataset = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dataset);

    // File hyperslab
    hsize_t file_offset[1] = { 0 };
    hsize_t file_count[1] = { n_cols };

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, file_offset, NULL, file_count, NULL);

    // Memory hyperslab
    hsize_t mem_offset[1] = { 0 };
    hsize_t mem_count[1] = { n_cols };

    hid_t memspace = H5Screate_simple(1, mem_count, NULL);
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, NULL, mem_count, NULL);

    // Write
    H5Dwrite(dataset, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, data);

    // Close handles
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dataset);
}
// Write 3D int data to the hdf5 file
static void write_3d(hid_t file_id, hsize_t n_x, hsize_t n_y, hsize_t n_z, int data[][N][N], const char* dataset_name) {
    // Identifier
    hid_t dataspace_id, dataset_id;
    hsize_t dims3d[3] = { n_x, n_y, n_z };

    dataspace_id = H5Screate_simple(3, dims3d, NULL);
    dataset_id = H5Dcreate(file_id, dataset_name, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Open dataset and get dataspace
    hid_t dataset = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dataset);

    // File hyperslab
    hsize_t file_offset[3] = { 0, 0, 0 };
    hsize_t file_count[3] = { n_x, n_y, n_z };

    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, file_offset, NULL, file_count, NULL);

    // Memory hyperslab
    hsize_t mem_offset[3] = { 0, 0, 0 };
    hsize_t mem_count[3] = { n_x, n_y, n_z };

    hid_t memspace = H5Screate_simple(3, mem_count, NULL);
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, NULL, mem_count, NULL);

    // Write
    H5Dwrite(dataset, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, data);

    // Close handles
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dataset);
}
// Writes the data to the hdf5 file
static void write_data(const char* name, data_to_save& saved_data)
{
    // Identifiers
    herr_t status;
    hid_t file_id, group_id, dataspace_id, dataset_id, attribute_id;

    // Open file and write header
    file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    group_id = H5Gcreate(file_id, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Gclose(group_id);

    write_1d(file_id, saved_data.merged_with.size(), saved_data.merged_with.data(), "merged_with");
    write_1d(file_id, saved_data.z_merge.size(), saved_data.z_merge.data(), "z_merge");
    write_1d(file_id, saved_data.z_form.size(), saved_data.z_form.data(), "z_form");
    write_1d(file_id, saved_data.parent_cells_merged.size(), saved_data.parent_cells_merged.data(), "parent_cells_merged");
    write_1d(file_id, saved_data.cells_merged.size(), saved_data.cells_merged.data(), "cells_merged");
    write_1d(file_id, saved_data.counts.size(), saved_data.counts.data(), "counts");
    write_1d(file_id, saved_data.offsets.size(), saved_data.offsets.data(), "offsets");
    write_1d(file_id, saved_data.bubble_cells.size(), saved_data.bubble_cells.data(), "bubble_cells");
    write_1d(file_id, saved_data.HII_Z_count.size(), saved_data.HII_Z_count.data(), "x_HI");
    write_1d(file_id, saved_data.I_com.size(), saved_data.I_com.data(), "i_com");
    write_1d(file_id, saved_data.dr_com.size(), saved_data.dr_com.data(), "dr_com");
    write_1d(file_id, saved_data.eff_volume.size(), saved_data.eff_volume.data(), "effective_volume");
    write_1d(file_id, saved_data.eff_volume_z.size(), saved_data.eff_volume_z.data(), "effective_volume_z");
    write_3d(file_id, N, N, N, saved_data.cell_to_bubble, "cell_to_bubble");

    // Close file
    H5Fclose(file_id);
}
// List the different groups in the hdf5 file
void list_datasets(hid_t group_id, const char* group_name) {
    hsize_t num_objs;
    herr_t status;
    int i;

    status = H5Gget_num_objs(group_id, &num_objs);

    for (i = 0; i < num_objs; i++) {
        char obj_name[1024];
        H5Gget_objname_by_idx(group_id, (hsize_t)i, obj_name, (size_t)1024);
        int obj_type = H5Gget_objtype_by_idx(group_id, (hsize_t)i);

        if (obj_type == H5G_GROUP) {
            printf("Group: %s/%s\n", group_name, obj_name);
            hid_t subgroup_id = H5Gopen(group_id, obj_name, H5P_DEFAULT);
            list_datasets(subgroup_id, obj_name);
            H5Gclose(subgroup_id);
        }
        else if (obj_type == H5G_DATASET) {
            printf("Dataset: %s/%s\n", group_name, obj_name);
        }
    }
}
vector<int> traverse_tree(vector<int> bubbles_merging, int num, bubble_group& bubble_groups, data_to_save& ordered_data_to_save, vector<int> ordered_indices) {
    vector<int> new_groups;
    for (int i = 0; i < ordered_data_to_save.merged_with.size(); i++) {
        if (ordered_data_to_save.merged_with[i] == num) {
            new_groups.push_back(i);
        }
    }
    if (!new_groups.empty()) {
        bubbles_merging.insert(bubbles_merging.end(), new_groups.begin(), new_groups.end());
        for (int i: new_groups)
            bubbles_merging = traverse_tree(bubbles_merging, i, bubble_groups, ordered_data_to_save, ordered_indices);
    }
    return bubbles_merging;
}
// Get the center of mass for each bubble group
void get_center_of_mass(bubble_group& bubble_groups, data_to_save& ordered_data_to_save,vector<int> ordered_indices) {
    for (int i = 0;i< ordered_indices.size();i++) {
        float x=0;
        float y=0;
        float z=0;
        float x2 = 0;
        float y2 = 0;
        float z2 = 0;
        float xy = 0;
        float xz = 0;
        float yz = 0;
        Coordinate starting_point;
        starting_point.x = bubble_groups.expansion_points_coords[ordered_indices[i]][0];
        starting_point.y = bubble_groups.expansion_points_coords[ordered_indices[i]][1];
        starting_point.z = bubble_groups.expansion_points_coords[ordered_indices[i]][2];

        vector<int> merged_groups;
        merged_groups = traverse_tree(merged_groups, i, bubble_groups, ordered_data_to_save, ordered_indices);
        merged_groups.push_back(i);
        for (int k : merged_groups) {
            int val = 0;
            if (k == i) {
                val = 3;
            }
            for (int j = val; j < bubble_groups.expansion_points_coords[ordered_indices[k]].size(); j += 3) {
                int distance_x = bubble_groups.expansion_points_coords[ordered_indices[k]][j] - starting_point.x;
                if (distance_x > N / 2) { distance_x -= N; }
                if (distance_x < -N / 2) { distance_x += N; }
                int distance_y = bubble_groups.expansion_points_coords[ordered_indices[k]][j + 1] - starting_point.y;
                if (distance_y > N / 2) { distance_y -= N; }
                if (distance_y < -N / 2) { distance_y += N; }
                int distance_z = bubble_groups.expansion_points_coords[ordered_indices[k]][j + 2] - starting_point.z;
                if (distance_z > N / 2) { distance_z -= N; }
                if (distance_z < -N / 2) { distance_z += N; }
                x += distance_x;
                y += distance_y;
                z += distance_z;
                x2 += (distance_x * distance_x);
                y2 += (distance_y * distance_y);
                z2 += (distance_z * distance_z);
                xy += (distance_x * distance_y);
                xz += (distance_x * distance_z);
                yz += (distance_y * distance_z);

            }
        }
        ordered_data_to_save.dr_com.push_back(x);
        ordered_data_to_save.dr_com.push_back(y);
        ordered_data_to_save.dr_com.push_back(z);
        ordered_data_to_save.I_com.push_back(x2);
        ordered_data_to_save.I_com.push_back(y2);
        ordered_data_to_save.I_com.push_back(z2);
        ordered_data_to_save.I_com.push_back(xy);
        ordered_data_to_save.I_com.push_back(xz);
        ordered_data_to_save.I_com.push_back(yz);
    }
}
// Sorts and saves the data_to_save
void save_the_data(string smooth, string res, float zreion[][N][N], bubble_group& bubble_groups, data_to_save& saved_data) {
    // Get the last group that did not move and update the number of its cells
    int num = bubble_groups.non_merged_groups[0];
    saved_data.cells_merged[num] = N * N * N;
    saved_data.parent_cells_merged[num] = N * N * N;

    // Loop through the points that did not yet get expanded to (as code ended whenever there was only one)
    // One bubble left. Thus, added these points to that last bubble and updated the corresponding data
    float min = 1000;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++) {
                    if (saved_data.cell_to_bubble[i][j][k] == -1) {
                        saved_data.cell_to_bubble[i][j][k] = num;
                        bubble_groups.expansion_points_coords[num].push_back(i);
                        bubble_groups.expansion_points_coords[num].push_back(j);
                        bubble_groups.expansion_points_coords[num].push_back(k);
                        bubble_groups.size[num] += 1;
                        saved_data.counts[num] += 1;
                        saved_data.eff_volume.push_back(bubble_groups.size[num]);
                        saved_data.eff_volume_z.push_back(zreion[i][j][k]);
                        if (zreion[i][j][k] < min)
                            min = zreion[i][j][k];
                        for (int l = 0; l < saved_data.HII_Z_Values.size(); l++)
                            if (zreion[i][j][k] > saved_data.HII_Z_Values[l]) {
                                saved_data.HII_Z_count[l] += 1;
                                break;
                            }
                    }
                }

    saved_data.z_merge[num] = min;

    // Order the data based on how big the bubbles were.
    // The largest one goes to index 0, then index 1, ...
    std::vector<std::pair<int, int>> valueIndexPairs;
    for (int i = 0; i < saved_data.cells_merged.size(); ++i) {
        valueIndexPairs.emplace_back(saved_data.cells_merged[i], i);
    }

    sort(valueIndexPairs.begin(), valueIndexPairs.end(), std::greater<>());

    std::vector<int> orderedIndices;
    orderedIndices.reserve(saved_data.cells_merged.size());
    for (const auto& pair : valueIndexPairs) {
        orderedIndices.push_back(pair.second);
    }
    // Change the ordering of the saved data with the new ordering of bubbles 
    vector<int> merged_with_holder;
    merged_with_holder.resize(saved_data.merged_with.size());
    vector<float> z_merge_holder;
    z_merge_holder.resize(saved_data.z_merge.size());
    vector<float> z_form_holder;
    z_form_holder.resize(saved_data.z_form.size());
    vector<int> parent_cells_merged_holder;
    parent_cells_merged_holder.resize(saved_data.parent_cells_merged.size());
    vector<int> cells_merged_holder;
    cells_merged_holder.resize(saved_data.cells_merged.size());
    vector<int> counts_holder;
    counts_holder.resize(saved_data.counts.size());
    for (int i = 0; i < orderedIndices.size(); i++) {
        merged_with_holder[i] = saved_data.merged_with[orderedIndices[i]];
        z_merge_holder[i] = saved_data.z_merge[orderedIndices[i]];
        z_form_holder[i] = saved_data.z_form[orderedIndices[i]];
        parent_cells_merged_holder[i] = saved_data.parent_cells_merged[orderedIndices[i]];
        cells_merged_holder[i] = saved_data.cells_merged[orderedIndices[i]];
        counts_holder[i] = saved_data.counts[orderedIndices[i]];
        saved_data.bubble_cells.insert(saved_data.bubble_cells.end(), bubble_groups.expansion_points_coords[orderedIndices[i]].begin(), bubble_groups.expansion_points_coords[orderedIndices[i]].end());
    }
    saved_data.merged_with = merged_with_holder;
    merged_with_holder.clear();
    saved_data.z_merge = z_merge_holder;
    z_merge_holder.clear();
    saved_data.z_form = z_form_holder;
    z_form_holder.clear();
    saved_data.parent_cells_merged = parent_cells_merged_holder;
    parent_cells_merged_holder.clear();
    saved_data.cells_merged = cells_merged_holder;
    cells_merged_holder.clear();
    saved_data.counts = counts_holder;
    counts_holder.clear();
    
    vector<int> cumsum(saved_data.counts.size());
    std::partial_sum(saved_data.counts.begin(), saved_data.counts.end(), cumsum.begin());
    int total = 0;
    vector<int> holder = saved_data.merged_with;
    // Change the numbering of the saved data with the new ordering of bubbles
    vector<int> group_to_index;
    
    group_to_index.resize(orderedIndices.size());
    
    for (int i = 0; i < orderedIndices.size(); i++) {
        group_to_index[orderedIndices[i]] = i;
    }

    for (int i = 0; i < orderedIndices.size(); i++) {
        if (saved_data.merged_with[i] == -1)
            continue;
        holder[i] = group_to_index[saved_data.merged_with[i]];
    }
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
            for (int l = 0; l < N; l++) {
                saved_data.cell_to_bubble[j][k][l] = group_to_index[saved_data.cell_to_bubble[j][k][l]];
            }
   for (int i = 0; i < orderedIndices.size(); i++) {
       saved_data.offsets[i] = (cumsum[i] - saved_data.counts[i]) * 3;
       total += saved_data.counts[i];
    }
    saved_data.offsets[bubble_groups.total_group_num] = total*3;
    saved_data.merged_with = holder;
    holder.clear();

    // Update the HII_count to be the sum of the previous points
    vector<int> cumsum_2(saved_data.HII_Z_count.size());
    std::partial_sum(saved_data.HII_Z_count.begin(), saved_data.HII_Z_count.end(), cumsum_2.begin());
    for (int i = 0; i < saved_data.HII_Z_count.size(); i++) {
        saved_data.HII_Z_count[i] = N * N * N - cumsum_2[i];
    }
    std::cout << saved_data.cells_merged[0] << ' ' << total << ' ' << N * N * N << ' '<< saved_data.eff_volume.size() << std::endl;
    // Calculate the center of mass for r and r^2 for each bubble.
    get_center_of_mass(bubble_groups, saved_data,orderedIndices);

    // Save the data
    string file_name = smooth + "_" + res + "_tree_data.hdf5";
    write_data(file_name.c_str(), saved_data);
}
// Print how long each stage took
void print_time(std::chrono::time_point<std::chrono::high_resolution_clock> start, std::chrono::time_point<std::chrono::high_resolution_clock> end,double entire_run_time) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    long long total_seconds = duration / 1000000;
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int seconds = total_seconds % 60;
    int microseconds = duration % 1000000;

    float faction_time = duration / entire_run_time;
    float scale = std::pow(10.0f, 5);

    std::cout << std::setw(2) << std::setfill('0') << hours << ":"
              << std::setw(2) << std::setfill('0') << minutes << ":"
              << std::setw(2) << std::setfill('0') << seconds << ":"
              << std::setw(6) << std::setfill('0') << microseconds << " |       " << std::setprecision(5) << std::round(faction_time * scale) / scale << "        |" << std::endl;
}
int main()
{
    
    hid_t file_id, dataset;
    herr_t status;

    // Open File and z_reion data
    string path;
    string smooth = "0ckpc";
    string res = "512";
    path = "C:\\Users\\natha\\OneDrive\\Documents\\Thesan\\Thesan-1\\postprocessing\\smooth_renderings\\smooth_renderings_" + smooth + "_" + res + "\\z_reion.hdf5";
    std::cout << path << std::endl;
    file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file_id, "ReionizationRedshift", H5P_DEFAULT);

    // Read the dataset into the array
    float zreion[N][N][N];

    status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, zreion);
    H5Dclose(dataset);
    H5Fclose(file_id);

    // Get initial conditions
    auto initilization_start = std::chrono::high_resolution_clock::now();
    bubble_group bubble_groups = get_bubble_groups(zreion);
    data_to_save saved_data(bubble_groups.total_group_num);
    multimap<float, int, DecreasingOrder> expansion_queue;

    // Update data based on initial conditions and find starting neighbors and expansion queue
    for (int i : bubble_groups.non_merged_groups) {
        saved_data.z_form[i] = bubble_groups.values[i][0];
        saved_data.eff_volume.push_back(bubble_groups.values[i].size());
        for (int j = 0; j < bubble_groups.expansion_points_coords[i].size(); j += 3) {
            saved_data.cell_to_bubble[bubble_groups.expansion_points_coords[i][j]][bubble_groups.expansion_points_coords[i][j + 1]][bubble_groups.expansion_points_coords[i][j + 2]] = i;
            saved_data.eff_volume_z.push_back(bubble_groups.values[i][0]);
            for (int k = 0; j < saved_data.HII_Z_Values.size(); k++)
                if (bubble_groups.values[i][0] > saved_data.HII_Z_Values[k]) {
                    saved_data.HII_Z_count[k] += 1;
                    break;
                }
        }
        for (int j = 0; j < bubble_groups.expansion_points_coords[i].size(); j += 3) {
            vector<int>r{ bubble_groups.expansion_points_coords[i][j] ,bubble_groups.expansion_points_coords[i][j+1],bubble_groups.expansion_points_coords[i][j+2] };
            find_neighbors(r, i, zreion, bubble_groups, saved_data);
        }
        find_expansion_queue(i, bubble_groups, expansion_queue); 
    }
    auto initilization_end = std::chrono::high_resolution_clock::now();

    auto algorithm_start = std::chrono::high_resolution_clock::now();
    int epoch = 0;
    int z_index = 0;
    float z;
    // Keep expanding until only one group left
    while (bubble_groups.non_merged_groups.size() > 1) {
        expand(z, zreion, bubble_groups, saved_data, expansion_queue, epoch);

        if (epoch % 1000 == 0)
            std::cout << bubble_groups.non_merged_groups.size() << ' ' << z << ' ' << epoch << std::endl;
        epoch += 1;

    }
    auto algorithm_end = std::chrono::high_resolution_clock::now();

    // Save the Data from the expansion
    auto saved_data_start = std::chrono::high_resolution_clock::now();
    save_the_data(smooth, res, zreion, bubble_groups, saved_data);
    auto saved_data_end = std::chrono::high_resolution_clock::now();

    // Print out runtime
    auto entire_run_time = std::chrono::duration_cast<std::chrono::microseconds>(saved_data_end - initilization_start).count();
    std::cout << "+-------------------------------------------------------+" << std::endl;
    std::cout << "|                Hr:Mi:Sc:uSc    | Fraction Total Time  |" << std::endl;
    std::cout << "| Initialization: ";
    print_time(initilization_start,initilization_end,entire_run_time);
    std::cout << "| Algorithm:     ";
    print_time(algorithm_start,algorithm_end,entire_run_time);
    std::cout << "| Saving Data:   ";
    print_time(saved_data_start,saved_data_end,entire_run_time);
    std::cout << "+-------------------------------------------------------+" << std::endl;
    return 0;
}
