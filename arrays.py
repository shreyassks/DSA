from typing import List


def findMaxConsecutiveOnes(nums: List[int]) -> int:
    """
    https://leetcode.com/problems/max-consecutive-ones/
    Time - O(N), Space - O(1)
    :param nums:
    :return:
    """
    count = 0
    max_ones = -10000
    for i in range(len(nums)):
        if nums[i] == 1:
            count += 1
            max_ones = max(max_ones, count)
        else:
            count = 0
            max_ones = max(max_ones, count)
    return max_ones


def removeDuplicates_brute(nums: List[int]) -> int:
    """
    https://leetcode.com/problems/remove-duplicates-from-sorted-array/
    Time - O(NlogN) - set insertion + O(N) - for loop, Space - O(N)
    :param nums:
    :return:
    """
    visited = list(set(nums))

    for i in range(len(visited)):
        nums[i] = visited[i]
    print(nums)

    return len(visited)


def removeDuplicates_optimal(nums: List[int]) -> int:
    """
    https://leetcode.com/problems/remove-duplicates-from-sorted-array/
    Time - O(N) - for loop, Space - O(1)
    :param nums:
    :return:
    """

    pt1, pt2 = 0, 1
    unique = 0
    while pt2 < len(nums)-1:
        if nums[pt2] == nums[pt1]:
            pt2 += 1
        if nums[pt2] != nums[pt1]:
            unique += 1
            pt1 += 1
            nums[pt1] = nums[pt2]
    print(nums)
    return unique + 1


def replace_elements_by_sign(array: List[int]) -> List[int]:
    """
    Takes Time complexity - O(N), Space complexity - O(N)
    :param array:
    :return:
    """
    replaced_arr = [0] * len(array)
    pos_ind, neg_ind = 0, 1

    for num in array:
        if num > 0:
            replaced_arr[pos_ind] = num
            pos_ind += 2
        else:
            replaced_arr[neg_ind] = num
            neg_ind += 2

    return replaced_arr


def longest_sub_array_with_sum_k_brute(array: List[int], k: int) -> int:
    """
    Time Complexity - O(n2), Space Complexity - O(1)
    :param array:
    :param k:
    :return:
    """
    ans = 0
    for i in range(len(array)):
        for j in range(i, len(array)):
            s = sum(array[i:j])
            if s == k:
                ans = max(ans, (j - i))
    return ans


def longest_sub_array_with_sum_k_optimal(array: List[int], k: int) -> int:
    """
    Time Complexity - O(n), Space Complexity - O(1)
    :param array:
    :param k:
    :return:
    """
    ans = 0
    left, right = 0, 0
    while right < len(array):
        s = sum(array[left:right + 1])
        right += 1
        if s == k:
            ans = max(ans, right - left)
        elif s > k:
            s -= array[left]
            left += 1
    return ans


def two_sum_better(array: List[int], k: int) -> List[int]:
    """
    Using hashmaps
    Time - O(NlogN), Space - O(k) + O(N)
    for loop - N, dict - logN
    :param array:
    :param k:
    :return:
    """
    lookup = {}
    ids = []
    for i, num in enumerate(array):
        diff = abs(array[i] - k)
        if diff in lookup:
            lookup_id = lookup[diff]
            ids.append(lookup_id)
            ids.append(i)
        lookup[array[i]] = i
    return ids


def two_sum_optimal(array: List[int], k: int) -> List:
    """
    Using Two pointers and sorting
    Time - O(N), Space - O(N)
    :param array:
    :param k:
    :return:
    """
    array.sort()
    ids = set()
    left, right = 0, len(array) - 1
    while left < right:
        s = array[left] + array[right]
        if s == k:
            ids.add((left, right))
            left += 1
            right -= 1
        elif s < k:
            left += 1
        else:
            right -= 1
    return list(ids)


def threeSum(nums: List[int]) -> List:
    """
    Two pointer approach with sorting
    Time - O(NlogN + N2), Space - O(1)
    :param nums:
    :return:
    """
    size = len(nums)-1
    three_sum = set()

    nums.sort()
    print(nums)
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        pt1 = i + 1
        pt2 = size
        while pt1 < pt2:
            s = nums[i] + nums[pt1] + nums[pt2]
            if s < 0:
                pt1 += 1
            elif s > 0:
                pt2 -= 1
            else:
                three_sum.add((nums[i], nums[pt1], nums[pt2]))
                pt1 += 1
                pt2 -= 1
    return list(three_sum)


def fourSum(nums: List[int], target: int) -> List:
    """
    Two pointer approach with sorting
    Time - O(N2 * N) = O(N3), Space - O(1)
    :param nums:
    :param target:
    :return:
    """
    size = len(nums)-1
    four_sum = set()

    nums.sort()
    print(nums)
    for i in range(0, len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        for j in range(i+1, len(nums)):
            if j != i+1 and nums[j] == nums[j-1]:
                continue
            pt1 = j + 1
            pt2 = size
            while pt1 < pt2:
                s = nums[i] + nums[j] + nums[pt1] + nums[pt2]
                if s < target:
                    pt1 += 1
                elif s > target:
                    pt2 -= 1
                else:
                    four_sum.add((nums[i], nums[j], nums[pt1], nums[pt2]))
                    pt1 += 1
                    pt2 -= 1
    return list(four_sum)


def longest_consecutive_sequence(array: List[int]) -> int:
    longest = 1
    filtered_array = set(array)
    for num in filtered_array:
        count = 1
        if (num - 1) not in filtered_array:
            next_num = num + 1
            while next_num in filtered_array:
                count += 1
                next_num += 1
        longest = max(longest, count)
    return longest


def setZeroes_brute(matrix: List[List[int]]) -> None:
    """
    if found a value zero, set all values in that row and column other than 0's to -100.5
    then iterate full array and set all -100.5 values to zero
    Time - O(n*m*(n+m)) = O(N3)
    Do not return anything, modify matrix in-place instead.
    """
    rows = len(matrix)
    columns = len(matrix[0])

    def rows_zero(idx):
        for h in range(columns):
            if matrix[idx][h] != 0:
                matrix[idx][h] = -100.5

    def columns_zero(idx):
        for h in range(rows):
            if matrix[h][idx] != 0:
                matrix[h][idx] = -100.5

    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] == 0:
                rows_zero(i)
                columns_zero(j)

    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] == -100.5:
                matrix[i][j] = 0


def setZeroes_better(matrix: List[List[int]]) -> None:
    """
    Time - O(2*n*m), Space - O(n)+O(m) for storing row and col ids
    Do not return anything, modify matrix in-place instead.
    """
    rows = len(matrix)
    columns = len(matrix[0])

    row_ids = [0]*rows
    col_ids = [0]*columns

    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] == 0:
                row_ids[i] = 1
                col_ids[j] = 1

    for i in range(rows):
        for j in range(columns):
            if row_ids[i] == 1 or col_ids[j] == 1:
                matrix[i][j] = 0


def trapping_rain_water_brute(height: List[int]) -> int:
    """
    https://leetcode.com/problems/trapping-rain-water/
    Time - O(N2), Space - O(1)
    :param height:
    :return:
    """
    water = 0
    for i, num in enumerate(height):
        if i < len(height) - 1:
            left = max(height[:i + 1])
            right = max(height[i + 1:])
            diff = min(left, right) - num
            if diff > 0:
                water += diff
    return water


def trapping_rain_water_optimal(height: List[int]) -> int:
    """
    https://leetcode.com/problems/trapping-rain-water/
    Time - O(N), Space - O(2N)
    :param height:
    :return:
    """
    water = 0
    prefix = []
    suffix = []

    m = -1000
    for i in range(len(height)):
        m = max(m, height[i])
        prefix.append(m)

    m = -1000
    for i in range(len(height)):
        m = max(m, height[::-1][i])
        suffix.append(m)

    print(prefix)
    print(suffix[::-1])
    for i, num in enumerate(height):
        if i < len(height) - 1:
            diff = min(prefix[i], suffix[::-1][i]) - num
            if diff > 0:
                water += diff
    return water


if __name__ == "__main__":

    arr = [1, 1, 1, 0, 0, 1]
    result = findMaxConsecutiveOnes(arr)
    print(f"Max consecutive ones: {result}")

    arr = [0, 0, 0, 1, 1, 2, 2, 4, 4]
    result = removeDuplicates_brute(arr)
    print(f"Remove duplicates brute: {result}")

    arr = [0, 0, 0, 1, 1, 2, 2, 4, 4]
    result = removeDuplicates_optimal(arr)
    print(f"Remove duplicates optimal: {result}")

    arr = [2, 3, 5, -1, -2, -3]
    result = replace_elements_by_sign(arr)
    print(f"Replace elements by sign : {result}")

    arr = [1, 2, 3, 1, 1, 1, 1, 4, 2, 3]
    result = longest_sub_array_with_sum_k_brute(arr, 9)
    print(f"Longest sub array with sum k brute: {result}")

    arr = [1, 2, 3, 1, 1, 1, 1, 4, 2, 3]
    result = longest_sub_array_with_sum_k_optimal(arr, 9)
    print(f"Longest sub array with sum k optimal: {result}")

    arr = [2, 6, 5, 8, 11]
    result = two_sum_better(arr, 16)
    print(f"Two Sum Better : {result}")

    arr = [2, 6, 5, 8, 11]
    result = two_sum_optimal(arr, 16)
    print(f"Two Sum Optimal : {result}")

    arr = [1, 1, 1, 2, 3, 2, 5, 6, 100, 103, 102, 4, 101, 104]
    result = longest_consecutive_sequence(arr)
    print(f"Longest consecutive sequence : {result}")

    # arr = [-1, 0, 1, 2, -1, -4]
    # arr = [0, 0, 0]
    arr = [-2, 0, 0, 2, 2]
    result = threeSum(arr)
    print(f"Three Sum Optimal: {result}")

    # arr = [1, 0, -1, 0, -2, 2]
    arr = [2, 2, 2, 2, 2]
    result = fourSum(arr, 8)
    print(f"Four Sum Optimal : {result}")

    arr = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    setZeroes_brute(arr)
    print(f"Set Zeros brute : {arr}")

    arr = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    setZeroes_better(arr)
    print(f"Set Zeros better : {arr}")

    arr = [0,1,0,2,1,0,1,3,2,1,2,1]
    result = trapping_rain_water_brute(arr)
    print(f"Trapping rain water brute: {result}")

    arr = [0,1,0,2,1,0,1,3,2,1,2,1]
    result = trapping_rain_water_optimal(arr)
    print(f"Trapping rain water optimal: {result}")
