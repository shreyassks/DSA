def factorial(n: int) -> int:
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


def sumOfDigits(s: int) -> int:
    if s == 0:
        return 0
    return s % 10 + sumOfDigits(s // 10)


def productOfDigits(s: int) -> int:
    if s < 10:
        return s
    return s % 10 * productOfDigits(s // 10)


def reverseNumber(n: int) -> int:
    if n % 10 == n:
        return n

    return int(f"{n % 10}{reverseNumber(n // 10)}")


def palindrome(n: int) -> bool:
    return n == reverseNumber(n)


class Solution:
    def helper1(self, num: int, c: int) -> int:
        if num == 0:
            return c

        if num % 10 == 0:
            c += 1
            return self.helper1(num // 10, c)
        return self.helper1(num // 10, c)

    def count_num_zeros(self, n: int) -> int:
        count = 0
        return self.helper1(n, count)

    def helper2(self, num: int, cnt: int) -> int:
        if num == 0:
            return cnt
        if num % 2 == 0:
            return self.helper2(num // 2, cnt + 1)
        else:
            return self.helper2(num - 1, cnt + 1)

    def num_steps_to_zero(self, num: int) -> int:
        cnt = 0
        return self.helper2(num, cnt)

    def helper3(self, arr: list[int], idx: int) -> bool:
        if idx == len(arr) - 1:
            return True
        return arr[idx] < arr[idx + 1] and self.helper3(arr, idx + 1)

    def sorted_array_recursion(self, arr: list[int]) -> bool:
        ids = 0
        return self.helper3(arr, ids)

    def helper4(self, arr: list[int], target: int, idx: int) -> int:
        if idx > len(arr) - 1:
            return -1
        if arr[idx] == target:
            return idx
        return self.helper4(arr, target, idx + 1)

    def linear_search(self, arr: list[int], target: int) -> int:
        return self.helper4(arr, target, 0)

    def helper5(self, arr: list[int], target: int, idx: int) -> int:
        if idx < 0:
            return -1
        if arr[idx] == target:
            return idx
        return self.helper5(arr, target, idx - 1)

    def linear_search_from_end(self, arr: list[int], target: int) -> int:
        idx = len(arr) - 1
        return self.helper5(arr, target, idx)

    def helper6(self, arr: list[int], target: int, idx: int, l: list[int]) -> list[int]:
        if idx < 0:
            return l
        if arr[idx] == target:
            l.append(idx)
        return self.helper6(arr, target, idx - 1, l)

    def linear_search_all_ids(self, arr: list[int], target: int) -> list[int]:
        idx = len(arr) - 1
        all_ids = []
        return self.helper6(arr, target, idx, all_ids)

    def helper7(self, arr: list[int], infected_plant_id: int, timestep: int) -> int:
        if len(arr) == 1:
            return timestep
        if infected_plant_id - 1 < 0:
            arr[infected_plant_id + 1] -= arr[infected_plant_id]
            timestep += 1
        elif infected_plant_id + 1 > len(arr)-1:
            arr[infected_plant_id - 1] -= arr[infected_plant_id]
            timestep += 1
        else:
            arr[infected_plant_id - 1] -= arr[infected_plant_id]
            arr[infected_plant_id + 1] -= arr[infected_plant_id]
            timestep += 1

        if arr[infected_plant_id - 1] < arr[infected_plant_id]:
            return self.helper7(arr[infected_plant_id+1:], 0, timestep)
        if arr[infected_plant_id + 1] < arr[infected_plant_id]:
            return self.helper7(arr[:infected_plant_id], len(arr)-1, timestep)

        return self.helper7(arr[:infected_plant_id], len(arr)-1, timestep) + self.helper7(arr[infected_plant_id+1:], 0, timestep)

    def infected_plants(self, arr: list[int], infected: int) -> int:
        steps = 0
        return self.helper7(arr, infected, steps)


# Find if an array is sorted or not
def sorted_array(arr: list[int]) -> bool:
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


def merge_intervals(arr: list[list]) -> list[list]:
    arr.sort()
    final = [arr[0]]
    for interval in arr[1:]:
        if final[0][0] <= interval[0] <= final[0][1]:
            final[0][1] = max(final[0][1], interval[1])
        else:
            final.append(interval)
    return final


if __name__ == "__main__":
    prob1 = factorial(6)
    print(f"Factorial - {prob1}")

    prob2 = sumOfDigits(12349)
    print(f"Sum of Digits - {prob2}")

    prob3 = productOfDigits(4567)
    print(f"Product of Digits - {prob3}")

    prob4 = reverseNumber(38495)
    print(f"Reverse Digits - {prob4}")

    prob5 = palindrome(73377)
    print(f"Palindrome - {prob5}")

    obj = Solution()
    prob6 = obj.count_num_zeros(102034)
    print(f"Number of Zeros - {prob6}")

    prob7 = obj.num_steps_to_zero(144)
    print(f"Number of steps to make zero - {prob7}")

    prob8 = sorted_array([1, 2, 33, 4, 7, 0])
    print(f"Is the array sorted - {prob8}")

    prob9 = obj.sorted_array_recursion([1, 2, 3, 4, 7])
    print(f"Is the array sorted with recursion - {prob9}")

    prob10 = obj.linear_search([1, 2, 33, 4, 7, 0], 2)
    print(f"Linear search - {prob10}")

    prob11 = obj.linear_search_from_end([1, 2, 2, 2, 7, 0], 2)
    print(f"Linear search from end - {prob11}")

    prob12 = obj.linear_search_all_ids([1, 2, 2, 2, 7, 0], 2)
    print(f"Linear search all indices of a value - {prob12}")

    # prob13 = obj.infected_plants([1, 3, 8, 6, 7, 4], 3)
    # print(f"Number of Time steps to become stable - {prob13}")
    array = [[1, 3], [2, 4], [6, 8], [9, 10]]
    array2 = [[6, 8], [1, 9], [2, 4], [4, 7]]
    prob14 = merge_intervals(array)
    print(f"Merged Intervals - {prob14}")
