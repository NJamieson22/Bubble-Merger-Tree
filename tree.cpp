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
#include <omp.h>
#define WRITE_ATTRIBUTE(attr_name, attr_value, attr_type)                                           \
  dataspace_id = H5Screate(H5S_SCALAR);                                                             \
  attribute_id = H5Acreate(group_id, attr_name, attr_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); \
  status       = H5Awrite(attribute_id, attr_type, &(attr_value));                                  \
  status       = H5Aclose(attribute_id);                                                            \
  status       = H5Sclose(dataspace_id);
const int N = 128;
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
    // Used for storing the points the bubble only expanded to
    vector<vector<int>>expansion_points_coords;
    // Vector of Unordered maps where each unordered map corresponds to its group. The keys are coordinates that neighbor the 
    // bubble groups coordinate, and the values are the corresponding values.
    // Used to easily check the uniqueness of neighboring points when merging
    vector<unordered_map<Coordinate, float, CoordinateHash>> neighboring_coords_and_points;
    // Vector of ordered multimaps where the keys are the values, and the values are the corresponding coordinate
    // Used to queue the next points of expansion
    vector<multimap<float, Coordinate, DecreasingOrder>> ordered_neighboring_values;
    // 2D vector where each vector represents the values of that bubble coordinates
    // Used to store all the initial value of the bubble group
    vector<vector<float>>values;
    // 1D vector where each index corresponds to that bubble group size
    vector<int> size;
    // 1D vector that decreases in size to represent bubbles that haven't merged into another one
    vector<int> non_merged_groups;
    // Int value to hold the total number of initial groups
    int total_group_num;
};
struct data_to_save {
    // For each 1D vector, the index corresponds to that group. So, index 5 is group 5.

    // Vector that stores what group each group merged with. For example, if group 3 merges with group 10,
    //The 3rd index of merged_with will have a value of 10.
    vector<int> merged_with;
    // Stores the z-value that group merged with another group.
    vector<float> z_merge;
    // Stores the z-value that the group formed.
    vector<float> z_form;
    // Stores the number of cells the parent had when that group merged.
    vector<int> parent_cells_merged;
    // Stores the number of cells the group had when it merged.
    vector<int> cells_merged;
    // Stores the number of cells that group expanded into. Only includes expansion points, not merging points
    vector<int> counts;
    // Offsets to get the coordinates of each bubble
    vector<int> offsets;
    // Stores an array of coordinates for each bubble, and offsets are used to get the coordinates of each bubble.
    // 1D array where each grouping of three is one coordinate. So, indexes 0, 1, and 2 are all one coordinate
    vector<int> bubble_cells;
    // Stores the center of mass for each bubble
    vector<float> r_com;
    // Stores the x,y, and z components of the r_com
    vector<float> dr_com;
    // Stores the center of mass for r^2 for each bubble
    vector<float> r2_com;
    // Stores the number of ionized cells for that range of z-values (ranges in HII_Z_Values)
    // Cumsummed at the end to get the ionization history
    vector<int> HII_Z_count;
    // The ranges for calculating the number of ionized cells
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
    // Index to store which Z-value boundary for ionized cells is at
    int HII_index = 0;
    // At each coordinate, show which bubble group grew into it.
    int cell_to_bubble[N][N][N];
    data_to_save(int n) {
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

// This function takes in a 3D point r and finds all of the adjacent coordinates. 
// If corners are true, then it also computes the adjacent corners.
//The 3D space is closed, so for a sized N space, at the coordinate value N, it goes to the 0 index.
std::vector<int> get_surrounding_coordinates(vector<int> r, bool corners) {
    vector<int> coordinate;
    coordinate.reserve(3 * 26);
    int x_increase;
    int y_increase;
    int z_increase;
    // Get all surrounding points, including corners and skipping self.
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
                    // Do % N because the grid is closed
                    coordinate.push_back(x_increase % N);
                    coordinate.push_back(y_increase % N);
                    coordinate.push_back(z_increase % N);

                }
            }

        }
    }
    return coordinate;
}
// This function finds and returns the starting bubble groups.
// It does so by finding the coordinates in the dataset that are local maxima,
// then returns those as the bubble groups
bubble_group get_bubble_groups(float zreion[][N][N]) {
    bubble_group bubble_groups;
    int current_group_num = 0;
    bool is_greater = false;

    // Loop through the entire dataset to check maximas
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                float value = zreion[i][j][k];
                vector<int> r{ i,j,k };
                vector<int> coords_next = get_surrounding_coordinates(r, true); // get neighboring coordinates
                for (int l = 0; l < coords_next.size(); l += 3) {
                    // if a neighboring point is greater than, then don't include it in the bubble groups
                    if (zreion[coords_next[l]][coords_next[l + 1]][coords_next[l + 2]] >= value) {
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
                map:multimap<float, Coordinate, DecreasingOrder> orderedValues;
                bubble_groups.ordered_neighboring_values.push_back(orderedValues);


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
    for (int i = 0; i < coords.size(); i += 3) {
        Coordinate point = { coords[i] ,coords[i + 1],coords[i + 2] };

        // Check neighbors if it is not already a point in the bubble_group
        if (saved_data.cell_to_bubble[coords[i]][coords[i + 1]][coords[i + 2]] == -1) {
            // Check the unordered map to see if it is already
            bool is_not_in = bubble_groups.neighboring_coords_and_points[group].find(point) == bubble_groups.neighboring_coords_and_points[group].end();
            if (is_not_in) {
                bubble_groups.neighboring_coords_and_points[group][point] = zreion[coords[i]][coords[i + 1]][coords[i + 2]];
                bubble_groups.ordered_neighboring_values[group].emplace(zreion[coords[i]][coords[i + 1]][coords[i + 2]], point);
            }
        }
    }
}
// This function performs a binary search on the expansion queue to slot in the new largest neighboring value for that group.
void binary_search(int group, float value, vector<pair<float, int>>& expansion_queue) {
    // Get the starting points
    int size = expansion_queue.size();
    int low = 0;
    // It will always find a spot, so while (true) is fine.
    while (true) {
        int mid = low + (size - low) / 2;
        // Check the different conditions for how to iterate/insert the piece
        if (mid == 0) {
            auto it = expansion_queue.begin();
            // If the first value is more than the value inserting, want to put our value after the first element
            if (expansion_queue[0].first > value)
                ++it;
            pair<float, int> h{ value,group };
            expansion_queue.insert(it, h);
            break;
        }
        else if (mid == expansion_queue.size() - 1) {
            auto it = expansion_queue.begin() + mid;
            // If the last value is smaller than the first value, we want to insert it before.
            if (expansion_queue[0].first < value)
                --it;
            pair<float, int> h{ value,group };
            expansion_queue.push_back(h);
            break;
        }
        else if (expansion_queue[mid - 1].first > value && expansion_queue[mid + 1].first > value) {
            low = mid + 1;
        }
        else if (expansion_queue[mid - 1].first < value && expansion_queue[mid + 1].first < value) {
            size = mid - 1;
        }
        else {
            auto it = expansion_queue.begin() + mid;
            if (expansion_queue[mid].first > value)
                ++it;
            pair<float, int> h{ value,group };
            expansion_queue.insert(it, h);
            break;
        }
    }
}
// Get the expansion queue for a certain group
void find_expansion_queue(int group, bubble_group& bubble_groups, vector<float>& expansion_queue) {
    // Since ordered_neighboring_values os a map decreasing in value, just get the first element
    auto it = bubble_groups.ordered_neighboring_values[group].begin();
    expansion_queue[group] = it->first;
}
// Merges all groups that need to be merged. Used for the single-point expansion.
void merge_groups_single(vector<int> groups_to_merge, Coordinate point_to_merge, float current_z, float zreion[][N][N], bubble_group& bubble_groups, data_to_save& saved_data, vector<float>& expansion_queue, int epoch) {
    vector<std::pair<int, int>> group_sizes;
    group_sizes.reserve(groups_to_merge.size());
    for (int i : groups_to_merge) {
        group_sizes.emplace_back(bubble_groups.size[i], i);
    }

    sort(group_sizes.begin(), group_sizes.end(), std::greater<>());

    vector<int> ordered_groups;
    ordered_groups.reserve(groups_to_merge.size());
    for (const auto& pair : group_sizes) {
        ordered_groups.push_back(pair.second);
    }
    int larger_group = ordered_groups[0];
    bubble_groups.neighboring_coords_and_points[larger_group].erase(point_to_merge);
    auto it = bubble_groups.ordered_neighboring_values[larger_group].begin();
    while (it->first == current_z) {
        Coordinate check_point = it->second;
        if (check_point.x == point_to_merge.x && check_point.y == point_to_merge.y && check_point.z == point_to_merge.z)
            break;
        ++it;
        if (it == bubble_groups.ordered_neighboring_values[larger_group].end()) {
            break;
        }
    }
    bubble_groups.ordered_neighboring_values[larger_group].erase(it);
    saved_data.cell_to_bubble[point_to_merge.x][point_to_merge.y][point_to_merge.z] = larger_group;

    // Save the necessary saved_data for this group
    saved_data.counts[larger_group] += 1;

    // Update the group for this new coordinate
    bubble_groups.expansion_points_coords[larger_group].push_back(point_to_merge.x);
    bubble_groups.expansion_points_coords[larger_group].push_back(point_to_merge.y);
    bubble_groups.expansion_points_coords[larger_group].push_back(point_to_merge.z);
    bubble_groups.size[larger_group] += 1;

    for (int i = 1; i < ordered_groups.size(); i++) {
        int smaller_group = ordered_groups[i];
        if (smaller_group == larger_group)
            continue;
        bubble_groups.neighboring_coords_and_points[smaller_group].erase(point_to_merge);
        it = bubble_groups.ordered_neighboring_values[smaller_group].begin();
        while (it->first == current_z) {
            Coordinate check_point = it->second;
            if (check_point.x == point_to_merge.x && check_point.y == point_to_merge.y && check_point.z == point_to_merge.z)
                break;
            ++it;
            if (it == bubble_groups.ordered_neighboring_values[smaller_group].end()) {
                break;
            }
        }
        bubble_groups.ordered_neighboring_values[smaller_group].erase(it);

        // Update the necessary saved_data characteristics
        saved_data.merged_with[smaller_group] = larger_group;
        saved_data.z_merge[smaller_group] = current_z;
        saved_data.cells_merged[smaller_group] = bubble_groups.size[smaller_group];
        saved_data.parent_cells_merged[smaller_group] = bubble_groups.size[larger_group];

        bubble_groups.size[larger_group] += bubble_groups.size[smaller_group];

        // Put in the neighboring points. There is a possibility for duplicate points, so we need to check for those points before adding them in.
        vector<Coordinate> neighbors_coords_to_add;
        vector<float> neighbors_values_to_add;
        for (const auto& entry : bubble_groups.neighboring_coords_and_points[smaller_group]) {
            if (bubble_groups.neighboring_coords_and_points[larger_group].find(entry.first) == bubble_groups.neighboring_coords_and_points[larger_group].end()) {
                neighbors_coords_to_add.push_back(entry.first);
                neighbors_values_to_add.push_back(entry.second);
            }
        }
        for (int j = 0; j < neighbors_coords_to_add.size(); j++) {
            bubble_groups.neighboring_coords_and_points[larger_group][neighbors_coords_to_add[j]] = neighbors_values_to_add[j];
            bubble_groups.ordered_neighboring_values[larger_group].insert({ neighbors_values_to_add[j] , neighbors_coords_to_add[j] });
        }

        // Remove merged group
        auto group_to_remove = find(bubble_groups.non_merged_groups.begin(), bubble_groups.non_merged_groups.end(), smaller_group);
        if (group_to_remove != bubble_groups.non_merged_groups.end()) {
            bubble_groups.non_merged_groups.erase(group_to_remove);
        }
        expansion_queue[smaller_group] = 0;
    }
    vector<int> r{ point_to_merge.x,point_to_merge.y,point_to_merge.z };
    find_neighbors(r, larger_group, zreion, bubble_groups, saved_data);
    if (!bubble_groups.ordered_neighboring_values[larger_group].empty()) {
        find_expansion_queue(larger_group, bubble_groups, expansion_queue);
    }
}
// Merges all groups that need to be merged
void merge_groups(vector<int> groups_to_merge, vector<int> points_to_merge, float current_z, float zreion[][N][N], bubble_group& bubble_groups, data_to_save& saved_data, vector<float>& expansion_queue, int epoch) {
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
        // Update data if it hasn't been added yet.
        if (not_in) {
            for (int j = coord_pos; j < coord_pos + 3; j++)
                bubble_groups.expansion_points_coords[larger_group].push_back(points_to_merge[j]);
            saved_data.counts[larger_group] += 1;
            bubble_groups.size[larger_group] += 1;
            offset_points[larger_group] += 1;
        }

        // Merge the groups, increasing the size accordingly and combining the coords.
        // Need to check if the smaller group hasn't already merged, because it is possible to merge with
        // the large group many times.
        if (expansion_queue[smaller_group] != 0) {
            bubble_groups.size[larger_group] += bubble_groups.size[smaller_group];

          // Put in the neighboring points. There is a possibility of duplicate points, so we need to check for those points before adding them.
            vector<Coordinate> neighbors_coords_to_add;
            vector<float> neighbors_values_to_add;
            for (const auto& entry : bubble_groups.neighboring_coords_and_points[smaller_group]) {
                if (bubble_groups.neighboring_coords_and_points[larger_group].find(entry.first) == bubble_groups.neighboring_coords_and_points[larger_group].end()) {
                    neighbors_coords_to_add.push_back(entry.first);
                    neighbors_values_to_add.push_back(entry.second);
                }
            }

            for (int j = 0; j < neighbors_coords_to_add.size(); j++) {
                bubble_groups.neighboring_coords_and_points[larger_group][neighbors_coords_to_add[j]] = neighbors_values_to_add[j];
                bubble_groups.ordered_neighboring_values[larger_group].insert({ neighbors_values_to_add[j] , neighbors_coords_to_add[j] });
            }
        }

        // Remove the merged group from the list of expanding groups, get the neighbor of the new point, and update expansion_queue.
        //The larger group might merge with the smaller group many times, so check if the smaller group hasn't already merged. 
        vector<int>::iterator group_to_remove;
        if (expansion_queue[smaller_group] != 0) {
            group_to_remove = find(bubble_groups.non_merged_groups.begin(), bubble_groups.non_merged_groups.end(), smaller_group);
            bubble_groups.non_merged_groups.erase(group_to_remove);
        }
        expansion_queue[smaller_group] = 0;

        // Only need to check the neighbor for the new coordinate once.
        if (not_in) {
            vector<int> r{ points_to_merge[coord_pos],points_to_merge[coord_pos + 1],points_to_merge[coord_pos + 2] };
            find_neighbors(r, larger_group, zreion, bubble_groups, saved_data);
        }
        if (!bubble_groups.ordered_neighboring_values[larger_group].empty()) {
            find_expansion_queue(larger_group, bubble_groups, expansion_queue);
        }
    }
}
// Given a merging list, update it so we do not get any error. 
// I.e., if group 1 merges with 2 then 2 merges with 3. If 1 is greater than 2,
// 2 will no longer exist for the 2 and 3 merger. Thus, we replace the 2 with 1.
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
            // If the larger_group is already there, then skip it (as there is nothing to replace)
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
// Expands the bubble_groups for the single-point expansion algorithm
void single_expansion(float& z, float zreion[][N][N], bubble_group& bubble_groups, data_to_save& saved_data, vector<float>& expansion_queue, int epoch) {
    int group_to_expand;
    float max_z = 0;
    vector<int> merging_groups = {};
    vector<int> expanded_groups = {};
    for (int i = 0; i < bubble_groups.total_group_num; i++) {
        if (expansion_queue[i] > max_z) {
            max_z = expansion_queue[i];
            group_to_expand = i;
        }
    }
    z = max_z;
    auto it = bubble_groups.ordered_neighboring_values[group_to_expand].begin();
    Coordinate pos = it->second;
    vector<int> coord = { pos.x,pos.y,pos.z };
    vector<int> surronding_coords;
    surronding_coords = get_surrounding_coordinates(coord, true);

    for (int i = 0; i < surronding_coords.size(); i += 3) {
        if (saved_data.cell_to_bubble[surronding_coords[i]][surronding_coords[i + 1]][surronding_coords[i + 2]] != -1 && saved_data.cell_to_bubble[surronding_coords[i]][surronding_coords[i + 1]][surronding_coords[i + 2]] != group_to_expand) {
            bool already_there = false;
            int parent = saved_data.cell_to_bubble[surronding_coords[i]][surronding_coords[i + 1]][surronding_coords[i + 2]];
            while (expansion_queue[parent] == 0) {
                parent = saved_data.merged_with[parent];
            }
            if (parent == group_to_expand)
                continue;
            for (int j : merging_groups) {
                if (parent == j) {
                    already_there = true;
                    break;
                }
            }
            if (already_there) {
                continue;
            }

            merging_groups.push_back(parent);
        }
    }
    while (max_z <= saved_data.HII_Z_Values[saved_data.HII_index])
        saved_data.HII_index += 1;
    saved_data.HII_Z_count[saved_data.HII_index] += 1;

    if (merging_groups.size() > 0) {
        merging_groups.push_back(group_to_expand);
        merge_groups_single(merging_groups, pos, max_z, zreion, bubble_groups, saved_data, expansion_queue, epoch);
    }
    else {
        bubble_groups.neighboring_coords_and_points[group_to_expand].erase(pos);
        bubble_groups.ordered_neighboring_values[group_to_expand].erase(it);
        saved_data.cell_to_bubble[coord[0]][coord[1]][coord[2]] = group_to_expand;

        // Save the necessary saved_data for this group
        saved_data.counts[group_to_expand] += 1;

        // Update the group for this new coordinate
        bubble_groups.expansion_points_coords[group_to_expand].push_back(coord[0]);
        bubble_groups.expansion_points_coords[group_to_expand].push_back(coord[1]);
        bubble_groups.expansion_points_coords[group_to_expand].push_back(coord[2]);
        bubble_groups.size[group_to_expand] += 1;

        //get neighbors of the new point and update the queue.
        find_neighbors(coord, group_to_expand, zreion, bubble_groups, saved_data);
        if (!bubble_groups.ordered_neighboring_values[group_to_expand].empty()) {
            find_expansion_queue(group_to_expand, bubble_groups, expansion_queue);
        }
    }
}
// Expands the bubble_groups for the biggest neighbor. If groups expand into the same space, it merges them.
void expand(float& z, float zreion[][N][N], bubble_group& bubble_groups, data_to_save& saved_data, vector<float>& expansion_queue, int epoch) {
    float max_z = 0;

    vector<int> expanded_groups = {};
    vector<int> non_merging_coords = {};
    vector<int> non_merging_groups = {};
    vector<int> merging_groups = {};
    vector<int> merging_coords = {};

    // Gets the biggest neighbor from the expansion_queue and the groups corresponding with it.
    #pragma omp parallel 
    {
        float local_max = 0;
        thread = omp_get_thread_num();
        n_threads = omp_get_num_threads();
        vector<pair<float, int>> threads_expanding_groups;

        #pragma omp for schedule(static)
        for (int i = 0; i < bubble_groups.total_group_num; i++) {
            if (expansion_queue[i] > local_max) {
                local_max = expansion_queue[i];
                pair<float, int> holder = { local_max,i };
                threads_expanding_groups = { holder };
            }
            else if (expansion_queue[i] == local_max)
                threads_expanding_groups.push_back({ local_max,i });
        }
        #pragma omp critical
        for (int i = 0; i < threads_expanding_groups.size(); i++) {
            if (threads_expanding_groups[i].first > max_z) {
                max_z = threads_expanding_groups[i].first;
                expanded_groups = { threads_expanding_groups[i].second };
            }
            else if (threads_expanding_groups[i].first == max_z) {
                expanded_groups.push_back(threads_expanding_groups[i].second);
            }
        }
    }
    z = max_z;
    // Loops through the groups to expand and expands/merges accordingly
    for (int i : expanded_groups) {

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

            // Check if the point has already been expanded to. If not, add it to itself
            // If yes, begin the process of merging
            if (saved_data.cell_to_bubble[position[j].x][position[j].y][position[j].z] == -1) {
                // Add data to the non_merging_groups and coords
                non_merging_groups.push_back(i);

                non_merging_coords.push_back(position[j].x);
                non_merging_coords.push_back(position[j].y);
                non_merging_coords.push_back(position[j].z);
                saved_data.cell_to_bubble[position[j].x][position[j].y][position[j].z] = i;
            }
            else {
                // Now starting the merging process. Since, for the the other group, the position was -1, we need to remove this from non_merging_groups. 
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

        //get neighbors of new points and update the queue.
        find_neighbors(coord, group, zreion, bubble_groups, saved_data);
        if (!bubble_groups.ordered_neighboring_values[group].empty()) {
            find_expansion_queue(group, bubble_groups, expansion_queue);
        }
    }
    // If we have merges, then merge the groups
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
// Actually writes the data to the hdf5 file
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
    write_1d(file_id, saved_data.r_com.size(), saved_data.r_com.data(), "r_com");
    write_1d(file_id, saved_data.r2_com.size(), saved_data.r2_com.data(), "r2_com");
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
// Get the center of mass for each bubble group
void get_center_of_mass(bubble_group& bubble_groups, data_to_save& ordered_data_to_save) {
    for (int i = 0; i < bubble_groups.total_group_num; i++) {
        float x=0;
        float y=0;
        float z=0;
        float r = 0;
        float r2 = 0;
        Coordinate starting_point;
        starting_point.x = bubble_groups.expansion_points_coords[i][0];
        starting_point.y = bubble_groups.expansion_points_coords[i][1];
        starting_point.z = bubble_groups.expansion_points_coords[i][2];
        for (int j = 3; j < bubble_groups.expansion_points_coords[i].size(); j += 3) {
            int distance_x = starting_point.x - bubble_groups.expansion_points_coords[i][j];
            if (distance_x > N / 2) { distance_x = N - distance_x; }
            int distance_y = starting_point.y - bubble_groups.expansion_points_coords[i][j + 1];
            if (distance_y > N / 2) { distance_y = N - distance_y; }
            int distance_z = starting_point.z - bubble_groups.expansion_points_coords[i][j + 2];
            if (distance_z > N / 2) { distance_z = N - distance_z; }
            x += distance_x;
            y += distance_y;
            z += distance_z;
            distance_x *= distance_x;
            distance_y *= distance_y;
            distance_z *= distance_z;
            r += sqrtf(distance_x + distance_y + distance_z);
            r2 += distance_x + distance_y + distance_z;

        }
        ordered_data_to_save.r_com.push_back(r / bubble_groups.expansion_points_coords[i].size() / 3);
        ordered_data_to_save.dr_com.push_back(x / bubble_groups.expansion_points_coords[i].size() / 3);
        ordered_data_to_save.dr_com.push_back(y / bubble_groups.expansion_points_coords[i].size() / 3);
        ordered_data_to_save.dr_com.push_back(z / bubble_groups.expansion_points_coords[i].size() / 3);
        ordered_data_to_save.r2_com.push_back(r2 / bubble_groups.expansion_points_coords[i].size() / 3);
    }
}
// Sorts and saves the data_to_save
void save_the_data(string smooth, string res, float zreion[][N][N], bubble_group& bubble_groups, data_to_save& saved_data) {
    // Get the last group that did not move and update the number of its cells
    int num = bubble_groups.non_merged_groups[0];
    saved_data.cells_merged[num] = N * N * N;
    saved_data.parent_cells_merged[num] = N * N * N;

    // Loop through the points that did not yet get expanded to (as code ended whenever there was only)
    // One bubble left. Thus, added this points to that last bubble and update the corresponding data
    float min = 1000;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) {
                if (saved_data.cell_to_bubble[i][j][k] == -1) {
                    saved_data.cell_to_bubble[i][j][k] = num;
                    bubble_groups.expansion_points_coords[num].push_back(i);
                    bubble_groups.expansion_points_coords[num].push_back(j);
                    bubble_groups.expansion_points_coords[num].push_back(k);
                    saved_data.counts[num] += 1;
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

    // Order the data based on how big the bubble were.
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
    data_to_save ordered_data_to_save(bubble_groups.total_group_num);
    for (int i = 0; i < orderedIndices.size(); i++) {
        ordered_data_to_save.merged_with[i] = saved_data.merged_with[orderedIndices[i]];
        ordered_data_to_save.z_merge[i] = saved_data.z_merge[orderedIndices[i]];
        ordered_data_to_save.z_form[i] = saved_data.z_form[orderedIndices[i]];
        ordered_data_to_save.parent_cells_merged[i] = saved_data.parent_cells_merged[orderedIndices[i]];
        ordered_data_to_save.cells_merged[i] = saved_data.cells_merged[orderedIndices[i]];
        ordered_data_to_save.counts[i] = saved_data.counts[orderedIndices[i]];
        ordered_data_to_save.bubble_cells.insert(ordered_data_to_save.bubble_cells.end(), bubble_groups.expansion_points_coords[orderedIndices[i]].begin(), bubble_groups.expansion_points_coords[orderedIndices[i]].end());
    }
    vector<int> cumsum(ordered_data_to_save.counts.size());
    partial_sum(ordered_data_to_save.counts.begin(), ordered_data_to_save.counts.end(), cumsum.begin());

    int total = 0;
    vector<int> holder = ordered_data_to_save.merged_with;
    // Change the numbering of the saved data with the new ordering of bubbles
    for (int i = 0; i < orderedIndices.size(); i++) {
        for (int j = 0; j < orderedIndices.size(); j++)
            if (ordered_data_to_save.merged_with[j] == orderedIndices[i])
                holder[j] = i;
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                for (int l = 0; l < N; l++)
                    if (saved_data.cell_to_bubble[j][k][l] == orderedIndices[i])
                        ordered_data_to_save.cell_to_bubble[j][k][l] = i;
        ordered_data_to_save.offsets[i] = (cumsum[i] - ordered_data_to_save.counts[i]) * 3;
        total += ordered_data_to_save.counts[i];
    }
    ordered_data_to_save.offsets[bubble_groups.total_group_num] = total;
    ordered_data_to_save.merged_with = holder;

    // Update the HII_count to be the sum of the previous points
    ordered_data_to_save.HII_Z_count = saved_data.HII_Z_count;
    vector<int> cumsum_2(ordered_data_to_save.HII_Z_count.size());
    partial_sum(ordered_data_to_save.HII_Z_count.begin(), ordered_data_to_save.HII_Z_count.end(), cumsum_2.begin());
    for (int i = 0; i < ordered_data_to_save.HII_Z_count.size(); i++) {
        ordered_data_to_save.HII_Z_count[i] = N * N * N - cumsum_2[i];
    }
    std::cout << ordered_data_to_save.cells_merged[0] << ' ' << total << ' ' << N * N * N << std::endl;

    // Calculate the center of mass for r and r^2 for each bubble.
    get_center_of_mass(bubble_groups, ordered_data_to_save);


    // Save the data
    string file_name = smooth + "_" + res + "_tree_data.hdf5";
    write_data(file_name.c_str(), ordered_data_to_save);
}

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
    string smooth = "125ckpc";
    string res = "128";
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
    auto initialization_start = std::chrono::high_resolution_clock::now();
    bubble_group bubble_groups = get_bubble_groups(zreion);
    data_to_save saved_data(bubble_groups.total_group_num);
    vector<float> expansion_queue(bubble_groups.total_group_num);
    

    // Update data based on initial conditions and find starting neighbors and expansion queue
    for (int i : bubble_groups.non_merged_groups) {
        saved_data.z_form[i] = bubble_groups.values[i][0];
        saved_data.cell_to_bubble[bubble_groups.expansion_points_coords[i][0]][bubble_groups.expansion_points_coords[i][1]][bubble_groups.expansion_points_coords[i][2]] = i;
        vector<int>r{ bubble_groups.expansion_points_coords[i][0] ,bubble_groups.expansion_points_coords[i][1],bubble_groups.expansion_points_coords[i][2] };
        find_neighbors(r, i, zreion, bubble_groups, saved_data);
        find_expansion_queue(i, bubble_groups, expansion_queue);
        for (int j = 0; j < saved_data.HII_Z_Values.size(); j++)
            if (bubble_groups.values[i][0] > saved_data.HII_Z_Values[j]) {
                saved_data.HII_Z_count[j] += 1;
                break;
            }
    }
    auto initialization_end = std::chrono::high_resolution_clock::now();
    
    int epoch = 0;
    int z_index = 0;
    float z;
    // Keep expanding until only one group left
    while (bubble_groups.non_merged_groups.size() > 1) {
        expand(z, zreion, bubble_groups, saved_data, expansion_queue, epoch);
        //expand(z, zreion, bubble_groups, saved_data, expansion_queue, epoch);
        if (epoch % 1000 == 0)
            std::cout << bubble_groups.non_merged_groups.size() << ' ' << z << ' ' << epoch << std::endl;
        epoch += 1;

    }
    auto algorithm_end = std::chrono::high_resolution_clock::now();

    // Save the Data from the expansion
    save_the_data(smooth, res, zreion, bubble_groups, saved_data);
    auto saved_data_end = std::chrono::high_resolution_clock::now();

    // Print out runtime
    auto entire_run_time = std::chrono::duration_cast<std::chrono::microseconds>(saved_data_end - initilization_start).count();
    std::cout << "+--------------------------------------------------------+" << std::endl;
    std::cout << "|                 Hr:Mi:Sc:uSc    | Fraction Total Time  |" << std::endl;
    std::cout << "| Initialization: ";
    print_time(initialization_start,initialization_end,entire_run_time);
    std::cout << "| Algorithm:      ";
    print_time(initialization_end,algorithm_end,entire_run_time);
    std::cout << "| Saving Data:    ";
    print_time(algorithm_end,saved_data_end,entire_run_time);
    std::cout << "+--------------------------------------------------------+" << std::endl;
  
    return 0;
}
