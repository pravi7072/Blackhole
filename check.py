
# This file contains a curated collection of DSA patterns focused on
# Arrays and Strings, along with their core concept explanations.

import math
from collections import defaultdict, Counter
import heapq

# ----------------------------------------------------------------------
# --- ARRAY & STRING PATTERNS ---
# ----------------------------------------------------------------------

# --- 1. Two Pointers Technique ---
# Core Concept: Use two pointers moving through data to solve problems in O(n) time instead of $O(n^{2})$[cite: 30].

def two_sum_ii(numbers, target):
    """
    Problem: Two Sum II - Input Array Is Sorted (Easy) [cite: 31]
    Given a sorted array, find two numbers that add up to target. Returns 1-indexed indices.
    """
    left, right = 0, len(numbers) - 1
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1] # 1-indexed [cite: 40]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

def three_sum(nums):
    """
    Problem: 3Sum (Medium) [cite: 56]
    Find all unique triplets that sum to zero.
    """
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        # Skip duplicates for first element [cite: 62]
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates for second and third elements [cite: 71]
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result

# --- 3. Sliding Window Pattern ---
# Core Concept: Maintain a window and slide it across the array/string to find optimal subarrays[cite: 140].

def length_of_longest_substring(s):
    """
    Problem: Longest Substring Without Repeating Characters (Medium) [cite: 141]
    Find the length of the longest substring without repeating characters.
    """
    char_map = {}
    left = 0
    max_length = 0
    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            left = char_map[s[right]] + 1 [cite: 149]
        char_map[s[right]] = right
        max_length = max(max_length, right - left + 1) [cite: 151]
    return max_length

def min_window(s, t):
    """
    Problem: Minimum Window Substring (Hard) [cite: 159]
    Find the minimum window substring of s that contains all characters of t.
    """
    if not s or not t:
        return ""
    
    dict_t = Counter(t) [cite: 166]
    required = len(dict_t) [cite: 167]
    left, formed = 0, 0 [cite: 168, 169]
    window_counts = {}
    ans = float("inf"), None, None
    
    for right in range(len(s)):
        character = s[right]
        window_counts[character] = window_counts.get(character, 0) + 1 [cite: 174]
        
        if character in dict_t and window_counts[character] == dict_t[character]:
            formed += 1 [cite: 176, 177]
            
        # Try to contract the window [cite: 178]
        while left <= right and formed == required:
            window_len = right - left + 1
            
            if window_len < ans[0]:
                ans = (window_len, left, right) [cite: 182, 183]
                
            character_left = s[left]
            window_counts[character_left] -= 1 [cite: 185]
            
            if character_left in dict_t and window_counts[character_left] < dict_t[character_left]:
                formed -= 1 [cite: 186, 187, 188]
            left += 1
        
    return s[ans[1]: ans[2] + 1] if ans[0] != float("inf") else "" [cite: 191]

# --- 5. Cyclic Sort ---
# Core Concept: Sort arrays containing numbers in a given range by placing each number at its correct index[cite: 236].

def missing_number(nums):
    """
    Problem: Missing Number (Easy) [cite: 237]
    Find the missing number in an array containing n distinct numbers in range [0, n].
    """
    n = len(nums)
    i = 0
    while i < n:
        correct_index = nums[i] [cite: 244]
        # Only swap if number is in range [0, n-1] and not in place
        if correct_index < n and nums[i] != nums[correct_index]: [cite: 245]
            nums[i], nums[correct_index] = nums[correct_index], nums[i] [cite: 246]
        else:
            i += 1
            
    # Find the missing number [cite: 249]
    for i in range(n):
        if nums[i] != i: [cite: 251]
            return i
            
    return n # The missing number is n (i.e., range [0, n] means size n+1)

def find_disappeared_numbers(nums):
    """
    Problem: Find All Numbers Disappeared in an Array (Easy) [cite: 258]
    Find all numbers that are missing from array of n integers in range [1, n].
    """
    i = 0
    n = len(nums)
    while i < n:
        correct_index = nums[i] - 1 # adjust for 1-indexed [cite: 261, 264]
        if 0 <= correct_index < n and nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i] [cite: 267]
        else:
            i += 1
            
    missing_numbers = []
    for i in range(n):
        if nums[i] != i + 1: [cite: 272, 273]
            missing_numbers.append(i + 1) [cite: 274]
            
    return missing_numbers

# --- 9. HashMap & HashSet ---
# Core Concept: Use hash tables for $O(1)$ lookups, frequency counting, and pattern matching[cite: 396].

def two_sum(nums, target):
    """
    Problem: Two Sum (Easy) [cite: 397]
    Find two numbers in array that add up to target.
    """
    num_map = {}
    for i, num in enumerate(nums): [cite: 401]
        complement = target - num [cite: 402]
        if complement in num_map:
            return [num_map[complement], i] [cite: 403]
        num_map[num] = i [cite: 404]
    return []

def group_anagrams(strs):
    """
    Problem: Group Anagrams (Medium) [cite: 409]
    Group strings that are anagrams of each other.
    """
    anagram_map = defaultdict(list) [cite: 413]
    for s in strs:
        # Sort the string to get the key [cite: 415]
        key = "".join(sorted(s))
        anagram_map[key].append(s) [cite: 416]
        
    return list(anagram_map.values()) [cite: 417]

# --- 17. Bitwise XOR ---
# Core Concept: Use XOR properties ($a^{\wedge}a=0, a^{\wedge}0=a$) to find unique elements[cite: 732].

def single_number(nums):
    """
    Problem: Single Number (Easy) [cite: 733]
    Find the single number that appears only once. All other numbers appear twice. [cite: 735, 736]
    """
    result = 0 [cite: 737]
    for num in nums:
        result ^= num [cite: 739]
    return result

def single_number_iii(nums):
    """
    Problem: Single Number III (Medium) [cite: 745]
    Find two numbers that appear only once. All other numbers appear twice. [cite: 747, 748]
    """
    # Step 1: XOR all numbers to get xor_sum of the two unique numbers [cite: 749]
    xor_sum = 0
    for num in nums:
        xor_sum ^= num [cite: 752]
        
    # Step 2: Find rightmost set bit [cite: 753]
    rightmost_set_bit = xor_sum & (-xor_sum) [cite: 754]
    
    # Step 3: Divide numbers into two groups based on the rightmost set bit [cite: 755]
    num1, num2 = 0, 0 [cite: 756]
    for num in nums:
        if num & rightmost_set_bit: [cite: 758]
            num1 ^= num [cite: 760]
        else:
            num2 ^= num [cite: 761]
            
    return [num1, num2]

# --- 16. Modified Binary Search (Applied to Arrays) ---
# Core Concept: Apply binary search on sorted/rotated arrays or search spaces[cite: 681].

def search(nums, target):
    """
    Problem: Search in Rotated Sorted Array (Medium) [cite: 682]
    Search target in a rotated sorted array.
    """
    left, right = 0, len(nums) - 1
    
    while left <= right: [cite: 686]
        mid = (left + right) // 2 [cite: 687]
        
        if nums[mid] == target:
            return mid [cite: 688]
            
        # Check if left half is sorted [cite: 690]
        if nums[left] <= nums[mid]: [cite: 691]
            # Target is in the sorted left half
            if nums[left] <= target < nums[mid]: [cite: 692]
                right = mid - 1 # Corrected for loop logic
            else:
                left = mid + 1 [cite: 695]
        
        # Right half is sorted [cite: 697, 698]
        else:
            # Target is in the sorted right half
            if nums[mid] < target <= nums[right]: [cite: 699]
                left = mid + 1
            else:
                right = mid - 1 # Corrected for loop logic
                
    return -1 [cite: 703]

def min_eating_speed(piles, h):
    """
    Problem: Koko Eating Bananas (Medium) [cite: 710]
    Find minimum eating speed to finish all bananas within h hours.
    """
    def can_finish(speed): [cite: 714]
        hours = 0
        for pile in piles:
            hours += math.ceil(pile / speed) [cite: 717]
        return hours <= h [cite: 718]

    left, right = 1, max(piles) [cite: 719]
    
    while left < right: [cite: 720]
        mid = (left + right) // 2 [cite: 721]
        if can_finish(mid): [cite: 722]
            right = mid [cite: 723] # Potential answer, try smaller speed
        else:
            left = mid + 1 [cite: 725] # Too slow, need faster speed
            
    return left [cite: 726]

# --- 18. Top K Elements (Heap/Priority Queue) ---
# Core Concept: Use heaps to efficiently find K largest/smallest elements[cite: 767].

def find_kth_largest(nums, k):
    """
    Problem: Kth Largest Element in an Array (Medium) [cite: 768]
    Find the kth largest element in an unsorted array. Uses min-heap of size k. [cite: 772]
    """
    heap = [] [cite: 773]
    for num in nums: [cite: 774]
        heapq.heappush(heap, num) [cite: 775]
        if len(heap) > k: [cite: 776]
            heapq.heappop(heap) [cite: 777]
    return heap[0] [cite: 778]

def top_k_frequent(nums, k):
    """
    Problem: Top K Frequent Elements (Medium) [cite: 792]
    Find the k most frequent elements.
    """
    count = Counter(nums) [cite: 796]
    # Use heapq.nlargest for efficiency
    return heapq.nlargest(k, count.keys(), key=count.get) [cite: 798]

# --- 20. Greedy Algorithms ---
# Core Concept: Make locally optimal choices to find global optimum[cite: 856].

def can_jump(nums):
    """
    Problem: Jump Game (Medium) [cite: 857]
    Check if you can reach the last index. [cite: 860]
    """
    max_reach = 0 [cite: 861]
    last_index = len(nums) - 1
    
    for i, jump in enumerate(nums): [cite: 862]
        if i > max_reach: [cite: 863, 864]
            return False
            
        max_reach = max(max_reach, i + jump) [cite: 866, 867]
        
        if max_reach >= last_index: [cite: 868, 869]
            return True
            
    return False

def can_complete_circuit(gas, cost):
    """
    Problem: Gas Station (Medium) [cite: 875]
    Find the starting gas station index to complete the circuit.
    """
    total_tank = 0 [cite: 878]
    curr_tank = 0 [cite: 879]
    start_station = 0 [cite: 880]
    
    for i in range(len(gas)): [cite: 881]
        net_gas = gas[i] - cost[i] [cite: 883, 885]
        total_tank += net_gas
        curr_tank += net_gas
        
        # If current tank is negative, reset starting station [cite: 886]
        if curr_tank < 0: [cite: 887]
            start_station = i + 1 [cite: 888]
            curr_tank = 0 [cite: 889]
            
    return start_station if total_tank >= 0 else -1 [cite: 890]

# --- 21. Dynamic Programming - 0/1 Knapsack (Applied to Arrays) ---
# Core Concept: Choose to include/exclude items to maximize value within constraints[cite: 895].

def can_partition(nums):
    """
    Problem: Partition Equal Subset Sum (Medium) [cite: 896]
    Check if array can be partitioned into two subsets with equal sum.
    """
    total_sum = sum(nums) [cite: 899]
    
    if total_sum % 2 != 0: [cite: 900, 901]
        return False

    target = total_sum // 2 [cite: 903]
    dp = [False] * (target + 1) [cite: 904]
    dp[0] = True [cite: 905]

    for num in nums: [cite: 906]
        # Iterate backwards to avoid using same number twice [cite: 907]
        for j in range(target, num - 1, -1): [cite: 907]
            dp[j] = dp[j] or dp[j - num] [cite: 907]

    return dp[target] [cite: 908]

def find_target_sum_ways(nums, target):
    """
    Problem: Target Sum (Medium) [cite: 913]
    Find number of ways to assign + or - to each number to reach target. [cite: 915]
    """
    total_sum = sum(nums) [cite: 916]
    
    if total_sum < abs(target) or (target + total_sum) % 2 == 1: [cite: 919]
        return 0

    # P = (target + total) / 2 (Subset sum to find) [cite: 924, 925]
    subset_sum = (target + total_sum) // 2
    dp = [0] * (subset_sum + 1) [cite: 926]
    dp[0] = 1 [cite: 927]

    for num in nums: [cite: 928]
        for j in range(subset_sum, num - 1, -1): [cite: 929]
            dp[j] += dp[j - num] [cite: 930]

    return dp[subset_sum] [cite: 931]

# --- 26. Ordered Set/Map (Applied to Arrays/Indexing) ---
# Core Concept: Maintain sorted order for efficient range queries and operations[cite: 1153].

def contains_nearby_almost_duplicate(nums, k, t):
    """
    Problem: Contains Duplicate III (Hard) [cite: 1154]
    Check if abs(i - j) <= k AND abs(nums[i] - nums[j]) <= t. [cite: 1157, 1158]
    Uses bucket sort concept. [cite: 1162]
    """
    if t < 0:
        return False [cite: 1159, 1160]
        
    w = t + 1 [cite: 1163]
    bucket = {} 
    
    def get_id(x): [cite: 1164]
        return x // w if x >= 0 else (x + 1) // w - 1 [cite: 1165]

    for i, num in enumerate(nums): [cite: 1166]
        bucket_id = get_id(num) [cite: 1167]
        
        # Check current bucket [cite: 1168]
        if bucket_id in bucket:
            return True [cite: 1169]
            
        # Check adjacent buckets [cite: 1170]
        if (bucket_id - 1) in bucket and abs(num - bucket[bucket_id - 1]) <= t: [cite: 1171]
            return True
            
        if (bucket_id + 1) in bucket and abs(num - bucket[bucket_id + 1]) <= t: [cite: 1173, 1174]
            return True
            
        bucket[bucket_id] = num [cite: 1176]
        
        # Remove element outside window [cite: 1177]
        if i >= k:
            del bucket[get_id(nums[i - k])] [cite: 1178]
            
    return False [cite: 1179]

# ----------------------------------------------------------------------
# --- TEST CASES (for self-verification) ---
# ----------------------------------------------------------------------

def run_tests():
    print("Running Array & String Pattern Tests...")
    
    # 1. Two Pointers
    assert two_sum_ii([2, 7, 11, 15], 9) == [1, 2] [cite: 47, 50]
    assert len(three_sum([-1, 0, 1, 2, -1, -4])) == 2
    
    # 3. Sliding Window
    assert length_of_longest_substring("abcabcbb") == 3 [cite: 154]
    assert min_window("ADOBECODEBANC", "ABC") == "BANC" [cite: 193]
    
    # 5. Cyclic Sort
    assert missing_number([3, 0, 1]) == 2 [cite: 255]
    assert find_disappeared_numbers([4, 3, 2, 7, 8, 2, 3, 1]) == [5, 6] [cite: 277]
    
    # 9. HashMap & HashSet
    assert two_sum([2, 7, 11, 15], 9) == [0, 1] [cite: 407]
    result_anagrams = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    assert len(result_anagrams) == 3 [cite: 419]
    
    # 17. Bitwise XOR
    assert single_number([2, 2, 1]) == 1 [cite: 742]
    result_single3 = single_number_iii([1, 2, 1, 3, 2, 5])
    assert set(result_single3) == {3, 5} [cite: 765]
    
    # 16. Modified Binary Search
    assert search([4, 5, 6, 7, 0, 1, 2], 0) == 4 [cite: 705, 706]
    assert min_eating_speed([3, 6, 7, 11], 8) == 4 [cite: 728, 729]
    
    # 18. Top K Elements
    assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5 [cite: 790]
    assert set(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == {1, 2} [cite: 807, 808]
    
    # 20. Greedy Algorithms
    assert can_jump([2, 3, 1, 1, 4]) == True [cite: 873]
    assert can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3 [cite: 892]
    
    # 21. Dynamic Programming
    assert can_partition([1, 5, 11, 5]) == True [cite: 911]
    assert find_target_sum_ways([1, 1, 1, 1, 1], 3) == 5 [cite: 933]
    
    # 26. Ordered Set/Map
    assert contains_nearby_almost_duplicate([1, 2, 3, 1], 3, 0) == True [cite: 1181, 1183, 1185]
    
    print("All Array & String Tests Passed. File ready for download. 🎉")

if __name__ == "__main__":
    # Uncomment to run the included tests:
    # run_tests() 
    pass