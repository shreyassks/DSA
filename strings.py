
def longestSubstringWithoutRepeatingCharacters(word: str) -> int:
    """
    https://leetcode.com/problems/longest-substring-without-repeating-characters
    Time - O(N), Space - O(1)
    :param word:
    :return:
    """
    longest = 0
    seen = set()
    left = 0

    for right in range(len(word)):
        while word[right] in seen:
            seen.remove(word[left])
            left += 1
        seen.add(word[right])
        longest = max(longest, right - left + 1)
    return longest


if __name__ == "__main__":
    # s = "abcabcbb"
    s = "bbbbbbb"
    result = longestSubstringWithoutRepeatingCharacters(s)
    print(result)