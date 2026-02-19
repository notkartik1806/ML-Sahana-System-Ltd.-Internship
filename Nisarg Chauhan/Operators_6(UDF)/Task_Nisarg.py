ls = []

# Geting number of elements from the user
n = int(input("Enter number of elements: "))

# Geting elements from the user

for i in range(n):
    element = int(input(f"Enter element {i+1}: "))
    ls.append(element)

print(f"Main list: {ls}")

total_sum = sum(ls)
print(f"Total sum: {total_sum}")


# Check if total sum is odd (cannot be partitioned equally)
if total_sum % 2 != 0:
    print("Equal partition not possible with given elements")
else:
    target = total_sum // 2
    
    # Using dynamic programming to check if subset with target sum exists
    dp = [False] * (target + 1)
    dp[0] = True  # Empty subset has sum 0
    
    # For each element in the list
    for num in ls:
        # Traverse from target to num (backwards to avoid using same element twice)
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    
    if dp[target]:
        # If partition is possible, find one valid partition using backtracking
        def find_partition(nums, target):
            def backtrack(index, current_subset, current_sum):
                if current_sum == target:
                    return current_subset[:]
                if index >= len(nums) or current_sum > target:
                    return None
                
                # Include current element
                current_subset.append(nums[index])
                result = backtrack(index + 1, current_subset, current_sum + nums[index])
                if result:
                    return result
                current_subset.pop()
                
                # Exclude current element
                result = backtrack(index + 1, current_subset, current_sum)
                if result:
                    return result
                
                return None
            
            return backtrack(0, [], 0)
        
        subset1 = find_partition(ls, target)
        if subset1:
            subset2 = [x for x in ls if x not in subset1 or subset1.count(x) < ls.count(x)]
            # Handle duplicates properly
            temp_ls = ls[:]
            for item in subset1:
                temp_ls.remove(item)
            subset2 = temp_ls
            
            print(f"\nSubset 1: {subset1} => {sum(subset1)}")
            print(f"Subset 2: {subset2} => {sum(subset2)}")
        else:
            print("Equal partition not found")
    else:
        print("Equal partition not possible with given elements")
