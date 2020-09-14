from typing import List
import collections

#2020.7.9
#判断是否是回文数字
def isPalindrome(self, x: int) -> bool:
    x = str(x)
    len_x = len(x)
    if len_x % 2 == 1:
        for i in range(0, int(len_x / 2)):
            if x[i] != x[len_x - i - 1]:
                return False
        return True
    elif len_x % 2 == 0:
        if len_x > 2:
            for i in range(0, int(len_x / 2)):
                if x[i] != x[len_x - i - 1]:
                    return False
            return True
        elif len_x == 2:
            if x[0] == x[1]:
                return True
            else:
                return False
    return False

#2020.7.10
#13.罗马数字转十进制数
def romanToInt(self, s: str) -> int:
    dic = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    len_s = len(s)
    Int = 0
    for index in range(0,len_s-1):
        if dic[s[index]]<dic[s[index+1]]:
            Int -= dic[s[index]]
        else:
            Int += dic[s[index]]
    return Int+dic[s[-1]]

#14.最长公共前缀
def longestCommonPrefix(self, strs: List[str]) -> str:
    # 判断语句中 None,  False, 空字符串"", 0, 空列表[], 空字典{}, 空元组()都相当于 False
    if not strs:
        return ""
    # 按照字母排序
    str0 = min(strs)
    str1 = max(strs)
    for i in range(len(str0)):
        if str0[i] != str1[i]:
            return str0[:i]
    return str0

def longestCommonPrefix2(self, strs):
    ans = ''
    # 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
    for i in zip(*strs): # i tuple元祖
        #set创建一个无序不重复元素集合
        if len(set(i)) == 1:
            ans += i[0]
        else:
            break
    return ans

#2020.7.11
#20.有效的括号(栈)
def isValid(self, s: str) -> bool:
    dic = {'(':')', '[':']', '{':'}', '?':'?'}
    stack = ['?']
    for i in s:
        if i in dic:
            stack.append(i)
        elif dic[stack.pop()] != i:
            return False
    return len(stack) == 1

#21.合并两个有序链表
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class LinkList:
    def __init__(self):
        self.head = None
    def initList(self, data):
        #创建头节点
        self.head = ListNode(data[0], None)
        r = self.head
        p = self.head
        for i in data[1:]:
            node = ListNode(i, None)
            p.next = node
            p = p.next
        return r
    def printList(self, head):
        if head == None:
            return
        node = head
        while node!=None:
            print(node.val, end=' ')
            node = node.next

def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    #两个链表本为升序
    if not l1: #如果l1为空
        return l2
    if not l2:
        return l1
    if l1.val <=l2.val: #递归
        l1.next = self.mergeTwoLists(l1.next, l2)
        return l1
    else:
        l2.next = self.mergeTwoLists(l1, l2.next)
        return l2

#2020.7.12
#26.删除排序数组中的重复项
def removeDuplicates(self, nums: List[int]) -> int: #输出删除重复项后的数组长度
    for i in range(len(nums)-1, 0, -1):
        if nums[i]==nums[i-1]:
            nums.pop(i)
    return len(nums)

def removeDuplicates2(self, nums: List[int]) -> int:
    if not nums:
        return 0
    i = 0
    for j in range(1, len(nums)):
        print(j)
        if nums[i]!=nums[j]:
            i += 1
            nums[i] = nums[j]
    return i+1

#23.移除元素
def removeElement(self, nums: List[int], val: int) -> int:
    for i in range(len(nums)-1,-1,-1):
        if nums[i]==val:
            nums.pop(i)
    return len(nums)

#28.实现strStr(),即needle在haystack中出现的位置
#字符串逐一比较
def strStr(self, haystack: str, needle: str) -> int:
    len_haystack, len_needle = len(haystack), len(needle)
    for start in range(len_haystack-len_needle+1):
        if haystack[start:start+len_needle] == needle:
            return start
    return -1
#双指针-----------------------------------------------------！
def strStr2(self, haystack: str, needle: str) -> int:
    len_haystack, len_needle = len(haystack), len(needle)
    if len_needle == 0:
        return 0
    p_haystack = 0
    while p_haystack < len_haystack-len_needle+1:
        #找到与needle第一个字符匹配的第一个字符
        while p_haystack < len_haystack-len_needle+1 and haystack[p_haystack] != needle[0]:
            p_haystack += 1
        curr_len = p_needle = 0
        while p_needle < len_needle and p_haystack < len_haystack and haystack[p_haystack] == needle[p_needle]:
            p_haystack += 1
            p_needle += 1
            curr_len += 1
        #匹配成功
        if curr_len == len_needle:
            return p_haystack-len_needle
        #不匹配，回溯
        p_haystack = p_haystack - curr_len + 1

#2020.7.13
#35.搜索插入位置
def searchInsert(self, nums: List[int], target: int) -> int:
    #二分查找
    len_nums = len(nums)
    low = 0
    high = len_nums-1
    while low <= high:
        mid = int((low+high)/2)
        if target>nums[mid]:
            low = mid+1
        elif target<nums[mid]:
            high = mid-1
        else:
            return mid
    return (mid if (target<nums[mid]) else mid+1)
#python
def searchInsert2(self, nums: List[int], target: int) -> int:
    nums.append(target)
    nums.sort()
    return nums.index(target)


#2020.7.14
#38.外观数列
def countAndSay(self, n: int) -> str:
    pre_result = '1'
    for i in range(1, n):
        next_result = ''
        num = pre_result[0]
        count = 1
        for j in range(1, len(pre_result)):
            if pre_result[j]==num:
                count += 1
            else:
                next_result += str(count) + num
                num = pre_result[j]
                count = 1
        next_result += str(count) + num
        pre_result = next_result
    return pre_result

#53.最大子序和
#给定一个整数数组，找到一个具有最大和的连续子数组（子数组最少包含一个元素）
def maxSubArray(self, nums: List[int]) -> int:
    pass

#2020.7.16
#58.最后一个单词的长度
#从前往后的找，效率较低
def lengthOfLastWord(self, s: str) -> int:
    if not s:
        return 0
    worlds_length=[]
    space_count = 0
    each_world_length = 0
    for i in range(0, len(s)):
        if s[i] != ' ':
            each_world_length += 1
        elif s[i] == ' ':
            space_count += 1
            if s[i-1] != ' ' or s[i-1] == None:
                worlds_length.append(each_world_length)
            each_world_length = 0
        if i==len(s)-1 and s[i] != ' ':
            worlds_length.append(each_world_length)
    if space_count != 0 and len(worlds_length) != 0:
        return worlds_length[-1]
    elif len(worlds_length) == 0:
        return 0
    else:
        return len(s)
#从后往前找，效率高
def lengthOfLastWord2(self, s: str) -> int:
    target= []
    len_s = len(s)
    while len_s>=1:#如果s非空
        if s[len_s-1] != ' ':
            target.append(s[len_s-1])
        elif target:#s[len_s-1]==' '&&target!=[]
            break
        len_s -= 1
    return len(target)
#利用python函数，代码执行时间长
def lengthOfLastWord3(self, s: str) -> int:
    if not s:
        return 0
    s_split = s.split()
    if not s_split:#如果s中只有空格
        return 0
    return len(s_split[-1])

#66.加一
def plusOne(self, digits: List[int]) -> List[int]:
    #从后往前算进位
    for i in range(len(digits)-1, -1, -1):
        if digits[i] != 9:
            digits[i] += 1
            return digits
        else:
            digits[i] = 0
            if digits[0] == 0:
                digits.insert(0, 1)
                return digits

#2020.7.18
#67.二进制求和（两个二进制字符串，返回它们的和（用二进制表示））
def addBinary(self, a: str, b: str) -> str:
    ab = list(str(int(a)+int(b)))
    ab.reverse()
    ab.append('0')
    for i in range(len(ab)):
        if int(ab[i]) >= 2:
            ab[i] = str(int(ab[i]) - 2)
            ab[i+1] = str(int(ab[i+1]) + 1)
    ab.reverse()
    value = ''
    if ab[0] == '0':
        for i in ab[1:len(ab)]:
            value += i
    else:
        for i in ab:
            value += i
    return value

#69.求x的平方根
#找到k^2<=num的最大k值
def mySqrt(self, x: int) -> int:
    low = 0
    high = x
    ans = -1
    while low <= high:
        mid = (low + high) // 2
        if mid*mid<=x:
            ans = mid
            low = mid+1
        else:
            high = mid - 1
    return ans

#70.爬楼梯，n阶到达楼顶，每次可以爬1/2个台阶，有多少种不同的爬楼方法
#递归
def climbStairs(self, n: int) -> int:
    #f(0)=1,f(1)=1,f(x)=f(x-1)+f(x-2)
    if n == 1:
        return 1
    if n == 2:
        return 2
    return self.climbStairs(None, n-1) + self.climbStairs(None, n-2)
#记忆化递归
def climbStairsMemo(n, memo:List[int]):
    if memo[n] > 0:
        return memo[n]
    if n == 1:
        memo[n] = 1
    elif n == 2:
        memo[n] = 2
    else:
        memo[n] = climbStairsMemo(n-1, memo) + climbStairsMemo(n-2, memo)
    return memo[n]
def climbStairs2(self, n: int) -> int:
    memo = []
    for i in range(n+1):
        memo.append(0)
    return climbStairsMemo(n, memo)
#动态规划
def climbStairs3(self, n: int) -> int:
    memo = {}
    memo[1] = 1
    memo[2] = 2
    for i in range(3, n+1):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[n]
#动态规划2
def climbStairs4(self, n: int) -> int:
    if n == 1:
        return 1
    elif n == 2:
        return 2
    else:
        a, b, temp = 1, 2, 0
        for i in range(3, n+1):
            temp = a + b
            a = b
            b = temp
        return temp

#83.删除排序链表中的重复元素
#定义链表结构
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next
#初始化链表
class LinkList:
    def __init__(self):
        self.head = None
    def initList(self, data):
        self.head = ListNode(data[0], None)
        r = self.head
        p = self.head
        for i in data[1:]:
            node = ListNode(i, None)
            p.next = node
            p = p.next
        return r
    def printList(self, head):
        if head == None:
            return
        node = head
        while node!=None:
            print(node.val, end=' ')
            node = node.next
#删除排序链表中的重复元素
def deleteDuplicates(self, head: ListNode) -> ListNode:
    if head == None or head.next == None:
        return head
    copyhead = ListNode(0)
    copyhead.next = head
    while head != None and head.next != None:
        if head.val == head.next.val:
            head.next = head.next.next
        else:
            head = head.next
    return copyhead.next

#88.合并两个有序数组
#双指针，从前往后
def merge(self, a:List[int], b:List[int]) -> List[int]:
    m = 0
    n = 0
    c = []
    while m <= (len(a)-1) and n <= (len(b)-1):
        if a[m] <= b[n]:
            c.append(a[m])
            m += 1
        else:
            c.append(b[n])
            n += 1
    #打印m,n
    if m == len(a):
        c[m+n:] = b[n:]
    else:
        c[m+n:] = a[m:]
    return c
#三指针，从后往前
def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    p = len(nums1) - 1
    p1 = m - 1
    p2 = n - 1
    while p1 >=0 and p2 >=0:
        if nums1[p1] >= nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
            p -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1
def merge2(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    if n == 0:
        pass
    elif m == 0:
        nums1[:n] = nums2[:n]
    else:
        a, b = m-1, n-1
        k = m+n-1
        while (a>=0) and (b>=0):
            if nums1[a]<=nums2[b]:
                nums1[k] = nums2[b]
                k -= 1
                b -= 1
            else:
                nums1[k] = nums1[a]
                k -= 1
                a -= 1
        if a>=0:
            pass
        if b>=0:
            nums1[0: k+1] = nums2[:b+1]

#2020.7.29
#100.相同的树(递归)
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    if p == None and q == None:
        return True
    if p == None and q != None:
        return False
    if p != None and q == None:
        return False
    if p.val != q.val:
        return False
    return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

#2020.7.30
#101.对称二叉树
#对称二叉树左子树先序遍历和右子树后序遍历的结果相反
def isSymmetric(self, root: TreeNode) -> bool:
    lli = []
    rli = []
    if root == None:
        return True
    if root and root.left == None and root.right == None:
        return True
    if root and root.left and root.right:
        self.pre_order(root.left, lli)
        self.post_order(root.right, rli)
        rli.reverse()
        if lli == rli:
            return True
        else:
            return False
#先序遍历左子树
def pre_order(self, root, li):
    if root:
        li.append(root.val)
        self.pre_order(root.left, li)
        self.pre_order(root.right, li)
    elif root == None:
        li.append(None)
#后序遍历右子树
def post_order(self, root, li):
    if root:
        self.post_order(root.left, li)
        self.post_order(root.right, li)
        li.append(root.val)
    elif root == None:
        li.append(None)

#104.二叉树的最大深度
#定义树节点
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

#深度优先+递归
def maxDepth1(self, root: TreeNode) -> int:
    if root == None:
        return 0
    else:
        return max(maxDepth1(root.left), maxDepth1(root.right))+1
#广度优先
def maxDepth2(self, root: TreeNode) -> int:
    if root == None:
        return 0
    counter = 1
    max_depth = 0
    nodelist = [(root, counter)]
    while nodelist:
        node, depth = nodelist.pop()
        if node.left:
            nodelist.append((node.left, depth+1))
        if node.right:
            nodelist.append((node.right, depth+1))
        if node.left == None and node.right == None:
            max_depth = max(max_depth, depth)
    return max_depth

#107.二叉树的层次遍历
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
    if root == None:
        return []
    queue = []
    nodelist = [root]
    while nodelist:
        nodelist_val = []
        next_node = []
        for node in nodelist:
            nodelist_val.append(node.val)
            if node.left:
                next_node.append(node.left)
            if node.right:
                next_node.append(node.right)
        #insert(location, object) location为插入的位置
        queue.insert(0, nodelist_val)
        nodelist = next_node
    return queue

#108.将有序数组转化为二叉搜索树
#该数组为二叉搜索树中序遍历的结果
def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    def make_tree(start_index, end_index):
        if start_index > end_index:
            return None
        mid_index = (start_index + end_index)//2
        this_tree_node = TreeNode(nums[mid_index])
        this_tree_node.left = make_tree(start_index, mid_index-1)
        this_tree_node.right = make_tree(mid_index+1, end_index)
        return this_tree_node
    return make_tree(0, len(nums) - 1)

#110.平衡二叉树
#递归
#自底向上
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def isBalanced(self, root: TreeNode) -> bool:
    return self.recurt(root) != -1
def recurt(self, root):
    if not root:#root为空
        return 0
    left = self.recurt(root.left)
    if left == -1:
        return -1
    right = self.recurt(root.right)
    if right == -1:
        return -1
    if abs(left - right) < 2:
        return max(left, right) +1
    else:
        return -1

#111.二叉树的最小深度
def minDepth(self, root: TreeNode) -> int:
    if not root:return 0
    def recursion(root1):
        if not root1:return float("inf")
        if not root1.left and not root1.right:return 1
        return min(recursion(root1.left), recursion(root1.right))+1
    return recursion(root)

#112.
#背
#DFS
def hasPathSum(self, root, sum) -> bool:
    if not root:
        return False
    if not root.left and not root.right:
        return sum == root.val
    return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)
#栈
def hasPathSum2(self, root, sum) -> bool:
    if not root:
        return False
    stack = [(root, root.val)]
    while stack:
        node, path = stack.pop()
        if not node.left and not node.right and path == sum:
            return True
        elif node.left:
            stack.append((node.left, path + node.left.val))
        elif node.right:
            stack.append((node.right, path + node.right.val))
        return False

#118.杨辉三角
def generate(self, numRows: int) -> List[List[int]]:
    if numRows == 0:
        return []
    nums = [[1]]
    while len(nums) < numRows:
        newrow = [a + b for a,b in zip([0] + nums[-1], nums[-1] + [0])]
        nums.append(newrow)
    return nums

#119.杨辉三角2
def getRow(self, rowIndex: int) -> List[int]:
    if rowIndex == 0:
        return [1]
    rows = [[1]]
    while len(rows) <= rowIndex:
        newrow = [a + b for a,b in zip([0] + rows[-1], rows[-1] + [0])]
        rows.append(newrow)
    return rows[rowIndex]

#121.买卖股票的最佳时机
def maxProfit(self, prices: List[int]) -> int:
    minprice = float('inf')
    maxprice = 0
    for price in prices:
        minprice = min(minprice, price)
        maxprofit = max(maxprofit, price - minprice)
    return maxprofit

##122.买卖股票的最佳时机2
def maxProfit(self, prices: List[int]) -> int:
    profits = 0
    for i in range(1, len(prices)):
        profit = prices[i] - prices[i - 1]
        if profit > 0: profits += profit
    return profits

#125.验证回文串
def isPalindrome(self, s: str) -> bool:
    strips = ''
    s = s.lower()
    for letter in s:
        if letter >= 'a' and letter<= 'z':
            strips += letter
        if letter >= 0 and letter <= 9:
            strips += letter
    if strips == strips[::-1]:
        return True
    return False

#136.之出现一次的数字
def singleNumber(self, nums: List[int]) -> int:
    return sum(set(nums))*2 - sum(nums)

def singleNumber2(self, nums: List[int]) -> int:
    ans = 0
    for num in nums:
        ans = num ^ ans #按位对比，相同取0，相异取1
    return ans

#141.环形链表
def hasCycle(self, head: ListNode) -> bool:
    a = set()
    while head:
        if head in a:
            return True
        a.add(head)
        head = head.next
    return False

#递归法
def hasCycle2(self, head: ListNode) -> bool:
    if not head:
        return False
    if head.val == 0xcafebabe:
        return True
    head.val = 0xcafebabe
    return self.hasCycle2(head.next)


#155.最小栈
import math
def __init__(self):
    self.stack = []
    self.min_stack = [math.inf]

def push(self, x: int) -> None:
    self.stack.append(x)
    self.min_stack.append(min(x, self.min_stack[-1]))
def pop(self) -> None:
    self.stack.pop()
    self.min_stack.pop()
def top(self) -> int:
    return self.stack[-1]
def getMin(self) -> int:
    return self.min_stack[-1]

#160.相交链表
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
    if not headA or not headB:
        return None
    nodea,nodeb = headA,headB
    while nodea != nodeb:
        nodea = nodea.next if nodea else headB
        nodeb = nodeb.next if nodeb else headA
    return nodea

#167.两数之和-输入有序数组
#哈希表
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    dict = {}
    for index, num in enumerate(numbers):
        if num in dict:
            return [dict[num]+1, index+1]
        dict[target - num] = index
#二分查找
#双指针
def twoSum2(self, numbers: List[int], target: int) -> List[int]:
    left,right = 0, len(numbers)-1
    while left <= right:
        s = numbers[left] + numbers[right]
        if s == target:
            return left + 1,right + 1
        if s < target:
            left += 1
        if s > target:
            right -= 1

#168.excel表列名称
def convertToTitle(self, n: int) -> str:
    res = ""
    while n:
        n -= 1
        n,y = divmod(n, 26)
        res = chr(y + 65) +res
    return res

#169.多数元素
def majorityElement(self, nums: List[int]) -> int:
    numsdic = {}
    hold = len(nums) // 2
    for num in nums:
        if num not in numsdic:
            numsdic[num] = 1
        else:
            numsdic[num] += 1
    for num in numsdic:
        if numsdic[num] > hold:
            return num
#摩尔投票法
def majorityElement(self, nums: List[int]) -> int:
    major = 0
    votes = 0
    for num in nums:
        if votes == 0:
            major = num
        if num == major:
            votes += 1
        if num != major:
            votes -= 1
    return major

#excel表列名称
def titleToNumber(self, s: str) -> int:
    d = len(s)
    value = 0
    sdict = {}
    for i in range(1, 27):
        sdict[chr(i + 64)] = i
    for i in s:
        if d > 0:#*乘法 **乘方
            value += sdict[i] * 26 ** (d - 1)
            d -= 1
    return value

#172.阶乘后的0
def trailingZeroes(self, n: int) -> int:
    # 返回 n/5 + n/5^2 + n/5^3 + ...
    if n == 0:
        return 0
    else:
        return n // 5 + self.trailingZeroes(n // 5)

#189.旋转数组
#暴力解法，容易超出时间限制
def rotate(self, nums: List[int], k: int) -> None:
    for i in range(0, k):
        prenum = nums[len(nums) -1]
        for j in range(0, len(nums)):
            temp = nums[j]
            nums[j] = prenum
            prenum = temp
    return nums
#三次反转
def rotate(self, nums: List[int], k: int) -> None:
    n = len(nums)
    k = k%n
    def swap(l, r):
        while (l<r):
            nums[l],nums[r] = nums[r],nums[l]
            l += 1
            r -= 1
    #第一段[0, n-k-1]
    #第二段反转[n-k, n-1]
    #全部反转all
    swap(0, n-k-1)
    swap(n-k, n-1)
    swap(0, n-1)

#190.颠倒二进制位（颠倒给定的32位无符号整数的二进制位）
def reverseBits(self, n: int) -> int:
    #int(x,base) base为是什么进制
    return int('0b' + bin(n)[2:].zfill(32)[::-1], 2)
#挪位置
def reverseBits2(self, n: int) -> int:
    ret, power = 0, 31
    #>>挪位 eg. a=60(0011 1100) -> a>>2 a=15(0000 1111)
    #& 参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0
    while n:
        # (n&1)检索最右边的位
        ret += (n & 1) << power
        #从右到左遍历输入的字符串
        n = n >> 1
        power -= 1
    return ret

#191.位1的个数
def hammingWeight(self, n: int) -> int:
    count = 0
    while n != 0:
        # n&(n-1)可消去n的二进制表示中的最后一个1
        n &= (n -1)
        count += 1
    return count

#193.有效电话号码（电话号码的形式位 （xxx）xxx-xxxx 或者 xxx-xxx-xxxx）
#Bash
# python3 -c "import re;lines=open('file.txt').readlines();lines=[i.strip()for i in lines];ans=[i for i in lines if re.match('^\(\d{3}\) \d{3}-\d{4}$',i) or re.match('^\d{3}-\d{3}-\d{4}$',i)];   print('\n'.join(ans))"

# import re;
# lines = open('file.txt').readlines()
# lines = [i.strip() for i in lines];
# ans = [i for i in lines if re.match('^\(\d{3}\) \d{3}-\d{4}$',i) or re.match('^\d{3}-\d{3}-\d{4}$',i)]
# print('\n'.join(ans))

#195.第十行
# import linecache
# ans = linecache.getline('file.txt',10).strip()
# print(ans)

#198.打家劫舍
def rob(self, nums: List[int]) -> int:
    if len(nums) == 0:
        return 0
    N = len(nums)
    dp = [0] * (N + 1)
    dp [0] = 0
    dp [1] = nums[0]
    for k in range(2, N+1):
        dp[k] = max(dp[k-1], nums[k-1]+dp[k-2])
    return dp[N]

#空间优化
def rob(self, nums: List[int]) -> int:
    prev = 0
    curr = 0
    for i in nums:
        prev, curr = curr, max(curr, prev+i)
    return curr

#202.快乐数
# 当输入2时报错
def isHappy(self, n: int) -> bool:
    if n == 1:
        return True
    num = 0
    for each in str(n):
        num += pow(int(each), 2)
    return self.isHappy(num)
def isHappy2(self, n: int) -> bool:
    visited = set() #visited判断循环结束
    while n != 1 and n not in visited:
        visited.add(n)
        next = 0
        while n!=0:
            next += (n%10) ** 2
            n //= 10
        n = next
    return n == 1

#203.删除链表元素
#定义节点
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
#生成链表
class LinkList:
    def __init__(self):
        self.head = None
    def initList(self, data):
        self.head = ListNode(data[0], None)
        r = self.head
        p = self.head
        for i in data[1:]:
            node = ListNode(i,None)
            p.next = node
            p = p.next
        return r
    def printList(self, head):
        if head == None:
            return
        node = head
        while node!=None:
            print(node.val, end=' ')
            node = node.next
def removeElements(self, head: ListNode, val: int) -> ListNode:
    sentinel = ListNode(0)
    sentinel.next = head
    prev,curr = sentinel, head
    while curr:
        if curr.val == val:
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
    return sentinel.next

#206.反转链表
def reverseList(self, head: ListNode) -> ListNode:
    pre = None
    curr = head
    while curr:
        temp = curr.next
        curr.next = pre
        pre = curr
        curr = temp
    return pre

#204.计数质数
def countPrimes(self, n: int) -> int:
    #统计所有小于非负整数n的质数的数量
    count = 0
    isprimes = [1] * n
    for i in range(2, n):
        #如果这个位置的flag为1，说明数字 i  没有被比 i 小的数整除过，说明它是质数
        if isprimes[i] == 1:
            count += 1
        j = i
        # 下面这几步的思路是， i 的倍数一定不是质数，将这些数的flag设置为0
        # 设置倍数 j ，初始化与 i 相等。 因为i也是一点点加上来的，比如 i=5的时候，i 的4倍一定在 i=4 时已经设置为0过
        while i*j<n:
            isprimes[i*j] = 0
            j += 1
    return count

def isIsomorphic(self, s, t):
    if not s:
        return True
    dic = {}
    for i in range(len(s)):
        if s[i] not in dic:
            if t[i] in dic.values():
                return False
            else:
                dic[s[i]] = t[i]
        else:
            if dic[s[i]] != t[i]:
                return False
    return True
#217.存在重复元素
def containsDuplicate(self, nums: List[int]) -> bool:
    if nums is not None:
        return False
    else:
        return len(nums) > len(set(nums))

#210.存在重复元素
def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    hash = {}
    for i in range(len(nums)):
        if nums[i] not in hash:
            hash[nums[i]] = i
        else:
            if (i - hash[nums[i]])<=k:
                return True
            else:
                hash[nums[i]] = i
    return False

#225.用队列实现栈
class MyStack:
    def __init__(self):
        self.q = []

    def push(self, x: int) -> None:
        self.q.append(x)
        q_length = len(self.q)
        #反转前n-1个元素
        while q_length>1:
            self.q.append(self.q.pop(0))
            q_length -= 1

    def pop(self) -> int:
        return self.q.pop(0)

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return not bool(self.q)
#226.翻转二叉树
def invertTree(self, root: TreeNode) -> TreeNode:
    #递归
    if not root:
        return
    root.left,root.right = root.right,root.left
    self.invertTree(root.left)
    self.invertTree(root.right)
    return root
#231.2的幂
#若n=2的x次幂，恒有n&(n-1)==0，因为n二进制最高位为1，其余为0；n-1最高位为0，其他为1
def isPowerOfTwo(self, n: int) -> bool:
    return n>0 and (n&(n - 1))==0

#232.用栈实现队列
class MyQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, x: int) -> None:
        self.stack1.append(x)

    def pop(self) -> int:
        while len(self.stack1) > 1:
            self.stack2.append(self.stack1.pop())
        element = self.stack1.pop() #默认弹出最后一个元素
        while len(self.stack2) > 0:
            self.stack1.append(self.stack2.pop())
        return element

    def peek(self) -> int:
        #得到第一个元素
        while len(self.stack1) > 1:
            self.stack2.append(self.stack1.pop())
        element = self.stack1.pop()
        self.stack2.append(element)
        while len(self.stack2) > 0:
            self.stack1.append(self.stack2.pop())
        return element

    def empty(self) -> bool:
        return len(self.stack1) == 0

#234.回文链表
def isPalindrome(self, head: ListNode) -> bool:
    stack = []
    curr = head
    while curr:
        stack.append(curr)
        curr = curr.next
    node1 = head
    while stack:
        node2 = stack.pop()
        if node1.val != node2.val:
            return False
        node1 = node1.next
    return True

#235.二叉搜索树的最近公共祖先
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root.val > p.val and root.val > q.val:
        return self.lowestCommonAncestor(root.left, p, q)
    elif root.val < p.val and root.val < q.val:
        return self.lowestCommonAncestor(root.right, p, q)
    else:
        return root

#237.删除链表中的节点
#非尾节点
def deleteNode(self, node):
    node.val = node.next.val
    node.next = node.next.next

#242.有效的字母异位词
def isAnagram(self, s: str, t: str) -> bool:
    dic1 = {}
    dic2 = {}
    for each1 in s:
        if each1 in dic1:
            dic1[each1] += dic1[each1]
        else:
            dic1[each1] = 1
    for each2 in t:
        if each2 in dic2:
            dic2[each2] += dic2[each2]
        else:
            dic2[each2] = 1
    if dic1 == dic2:
        return True
    else:
        return False
def isAnagram2(self, s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    se = set(s)
    if se == set(t):
        for i in se:
            if s.count(i) != t.count(i):return False
        return True
    else:
        return False

#257.二叉树的所有路径
#深度优先遍历
def binaryTreePaths(self, root: TreeNode) -> List[str]:
    if not root:
        return []
    res = []
    def dfs(root, path):
        path += str(root.val)
        if not root.left and not root.right:
            res.append(path)
        elif not root.left:
            dfs(root.right, path+'->')
        elif not root.right:
            dfs(root.left, path+'->')
        else:
            dfs(root.left, path+'->')
            dfs(root.right, path+'->')
    dfs(root, '')
    return res

#258.各位相加
def addDigits(self, num: int) -> int:
    while num > 9:
        num = eval('+'.join(n for n in str(num)))
    return num

#263.丑数
def isUgly(self, num: int) -> bool:
    if num == 1:
        return True
    if num < 1:
        return False
    if num % 2 == 0:
        num = num // 2
    elif num % 3 == 0:
        num = num // 3
    elif num % 5 == 0:
        num = num // 5
    else:
        return False
    return self.isUgly(num)
def isUgly2(self, num: int) -> bool:
    for p in 2,3,5:
        while num%p == 0<num:
            num/=p
#268.缺失数字
def missingNumber(self, nums: List[int]) -> int:
    nums.sort()
    # Ensure that n is at the last index
    if nums[-1] != len(nums):
        return len(nums)
    # Ensure that 0 is at the first index
    elif nums[0] != 0:
        return 0
    # If we get here, then the missing number is on the range (0, n)
    for i in range(1, len(nums)):
        expected_num = nums[i - 1] + 1
        if nums[i] != expected_num:
            return expected_num

#278.第一个错误的版本
def firstBadVersion(self, n):
    start = 0
    end = n-1
    while start<=end:
        mid = (start+end) // 2
        if isBadVersion(mid):
            end = mid - 1
        else:
            start = mid + 1
    return start

#283.移动零
def moveZeroes(self, nums: List[int]) -> None:
    i = 0
    for j in range(len(nums)):
        if nums[j] != 0:
            nums[i],nums[j] = nums[j],nums[i]
            i += 1
    return nums
def moveZeroes2(self, nums: List[int]) -> None:
    count = 0
    for i in range(len(nums)):
        if nums[i] == 0:
            count+=1
        elif count>0:
            nums[i-count],nums[i] = nums[i],0
    return nums

#290.单词规律
def wordPattern(self, pattern: str, str: str) -> bool:
    s = str.split(' ')
    if len(s) != len(pattern):
        return False
    dic = {}
    for i,x in enumerate(s):
        if pattern[i] not in dic:
            if x in dic.values():
                return False
            dic[pattern[i]] = x
        else:
            if x != dic[pattern[i]]:
                return False
    return True

#292.Nim游戏
def canWinNim(self, n: int) -> bool:
    #return (n&3)!=0
    return (n%4) != 0

#299.猜数字游戏
def getHint(self, secret: str, guess: str) -> str:
    A,B = 0,0
    dic1, dic2 = {},{}
    siz = len(secret)
    for i in range(siz):
        if guess[i] == secret[i]:
            A += 1
        else:
            if secret[i] not in dic1:
                dic1[secret[i]] = 1
            else:
                dic1[secret[i]] += 1
            if guess[i] not in dic2:
                dic2[guess[i]] = 1
            else:
                dic2[guess[i]] += 1
    for x in dic1:
        if x in dic2:
            B += min(dic1[x], dic2[x])
    return str(A) + 'A' + str(B) + 'B'


#303.区域检索-数组不可变
class NumArray:
    #缓存
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.sums = [0]
        for i in range(len(nums)):
            self.sums.append(nums[i] + self.sums[i])

    def sumRange(self, i: int, j: int) -> int:
        return self.sums[j+1] - self.sums[i]

#326.3的幂
def isPowerOfThree(self, n: int) -> bool:
    if n == 1:
        return True
    while n != 1 and n % 3 == 0:
        return self.isPowerOfThree(n/3)
    if n > 3 or n == 0:
        return False

#342.4的幂
def isPowerOfFour(self, num: int) -> bool:
    #100 10000 1000000 1后的0为偶数个
    #十进制转二进制
    if num <= 0:
        return False
    b = []
    while True:
        s = num // 2
        y = num % 2
        b = b + [y]
        if s == 0:
            break
        num = s
    if len(b) % 2 == 1 and sum(b) == 1:
        return True
    else:
        return False

#344.反转字符串
def reverseString(self, s: List[str]) -> None:
    s = s.reverse()

#345.反转字符串中的元音字母
def reverseVowels(self, s: str) -> str:
    worlds = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    low, high = 0, len(s) - 1
    s = list(s)
    while low <= high:
        while low < high and s[high] not in worlds:
            high -= 1
        while low < high and s[low] not in worlds:
            low += 1
        s[low], s[high] = s[high],s[low]
        low += 1
        high -= 1
    return "".join(s)

#349.两个数组的交集
def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    samenums = []
    nums1,nums2 = set(nums1),set(nums2)
    if len(nums1) <= len(nums2):
        for each in nums1:
            if each in nums2:
                samenums.append(each)
    else:
        for each in nums2:
            if each in nums1:
                samenums.append(each)
    return samenums

#350.两个数组的交集2
def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    samenums = []
    if len(nums1) <= len(nums2):
        for each in nums1:
            if each in nums2:
                samenums.append(each)
    else:
        for each in nums2:
            if each in nums1:
                samenums.append(each)
    return samenums

def intersect2(self, nums1: List[int], nums2: List[int]) -> List[int]:
    num1 = collections.Counter(nums1)
    num2 = collections.Counter(nums2)
    num = num1&num2
    return num.elements()

#367.有效的完全平方数
def isPerfectSquare(self, num: int) -> bool:
    l, r = 1, num
    while l < r:
        mid = (l + r)//2
        if mid * mid < num:
            l = mid + 1
        else:
            r = mid
    return l * l == num

#371.两整数之和
def getSum(self, a, b):
    #2^32
    MASK = 0x100000000
    #整型最大值,给int类型赋值的话，0X7FFFFFFF代表最大值，0X80000000代表最小值
    MAX_INT = 0x7FFFFFFF
    MIN_INT = MAX_INT + 1
    while b != 0:
        #计算进位
        carry = (a&b) << 1
        # 取余范围限制在 [0, 2^32-1] 范围内
        a = (a^b)%MASK
        b = carry%MASK
    return a if a<=MAX_INT else ~((a % MIN_INT) ^ MAX_INT)

#374.猜数字大小
# def guessNumber(self, n: int) -> int:
#     start = 0
#     end = n
#     while start <= end:
#         mid = (start + end) // 2
#         if guess(mid) == 0:
#             return mid
#         elif guess(mid) > 0:
#             start = mid + 1
#         else:
#             end = mid - 1

#383.赎金信
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        n = collections.Counter(ransomNote)
        return collections.Counter(magazine) & n == n

#387.
def firstUniqChar(self, s: str) -> int:
    Hash = {}
    for i in s:
        Hash[i] = Hash.get(i, 0) + 1
    for key in Hash.keys():
        if Hash[key] == 1:
            return s.find(key)
    return -1

#389.找不同
def findTheDifference(self, s: str, t: str) -> str:
    s1 = collections.Counter(s)
    t1 = collections.Counter(t)
    for key in t1:
        if t1.get(key, 0) != s1.get(key, 0):
            return key

#392.判断子序列
def isSubsequence(self, s: str, t: str) -> bool:
    list_s = []
    for i in s:
        list_s.append(i)
    for world in t:
        if world == list_s[-1]:
            list_s.pop()
        elif world in list_s and world != list_s[-1]:
            return False
    if len(list_s) == 0:
        return True

#迭代
def isSubsequence2(self, s: str, t: str) -> bool:
    if not s:
        return True
    for i in t:
        if s[0] == i:
            s = s[1:]
        if not s:
            return True
    return False

#401.二进制手表
def readBinaryWatch(self, num: int) -> List[str]:
    list1 = []
    #计算给定值的二进制
    def count1(n):
        res = 0
        while n != 0:
            n = n & (n - 1)
            res += 1
        return res
    for i in range(12):
        for j in range(60):
            if (count1(i) + count1(j) == num):
                if j < 10:
                    s = str(i) + ':0' + str(j)
                else:
                    s = str(i) + ':' + str(j)
                list1.append(s)
    return list1

#404.左叶子之和
#递归
def sumOfLeftLeaves(self, root: TreeNode) -> int:
    if not root:
        return 0
    if root and root.left and not root.left.left and not root.left.right:
        return root.left.val + self.sumOfLeftLeaves(root.right)
    return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

#迭代
def sumOfLeftLeaves2(self, root: TreeNode) -> int:
    if not root:
        return 0
    sum_ = 0
    ans = [root]
    while ans:
        r = ans.pop()
        if r.left and not r.left.left and not r.left.right:
            sum_ += r.left.val
        if r.left:
            ans.append(r.left)
        if r.right:
            ans.append(r.right)
    return sum_

#405.数字转化为十六进制数
def toHex(self, num: int) -> str:
    #如果小于0，进行补码
    max_int = 0xffffffff + 0x00000001
    if num < 0:
        num += max_int
    hex_convert = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f')
    divisor = 0x00000010
    result = ''
    #按照位运算取模，并构造结果字符串
    quotient, remainder = divmod(num, divisor)
    result = hex_convert[remainder] + result
    while True:
        quotient, remainder = divmod(quotient, divisor)
        if not (quotient == remainder == 0):
            result = hex_convert[remainder] + result
        else:
            break
    return result

#最长回文串
def longestPalindrome(self, s: str) -> int:
    count = collections.Counter(s).values()
    x = sum([item // 2*2 for item in count if (item//2 > 0)])
    return x if x == len(s) else x+1

#412.Fizz Buzz
def fizzBuzz(self, n: int) -> List[str]:
    list1 = []
    for i in range(0, n + 1):
        if i % 15 == 0:
            list1.append("FizzBuzz")
        elif i % 3 == 0:
            list1.append("Fizz")
        elif i % 5 == 0:
            list1.append("Buzz")
        else:
            list1.append(str(i))
    return list1

def fizzBuzz2(self, n: int) -> List[str]:
    list1 = []
    num_tuple_list = [(15, 'FizzBuzz'), (3, 'Fizz'), (5, 'Buzz'), (1, '')]
    for num in range(1, n + 1):
        for div, char in num_tuple_list:
            if num % div == 0:
                list1.append(str(num) if div == 1 else char)
                break
    return list1

#414.第三大的数
def thirdMax(self, nums: List[int]) -> int:
    nums = list(set(nums))
    a = b = c = float('-inf')
    for num in nums:
        if num > a:
            c = b
            b = a
            a = num
        elif num > b:
            c = b
            b = num
        elif num > c:
            c = num
    #如果c是负无穷
    return a if isinf(c) else c

def thirdMax2(self, nums: List[int]) -> int:
    first = second = third = float('-inf')
    for num in nums:
        if num > third:
            if num < second:
                third = num
            elif num > second:
                if num < first:
                    third = second
                    second = num
                elif num > first:
                    third = second
                    second = first
                    first = num
    if third == float('-inf'):
        return first
    else:
        return third

#415.字符串相加
def addStrings(self, num1: str, num2: str) -> str:
    res = ""
    i, j, carry = len(num1) - 1, len(num2) - 1, 0
    while i >= 0 or j >= 0:
        n1 = int(num1[i]) if i >= 0 else 0
        n2 = int(num2[j]) if j >= 0 else 0
        temp = n1 + n2 + carry
        carry = temp // 10
        res = str(temp % 10) + res
        i, j = i - 1, j - 1
    return "1" + res if carry else res

#字符串中的单词数
def countSegments(self, s: str) -> int:
    if s == '':
        return 0
    res = 0
    s = s + ' '
    k = 0
    for i in s:
        if k == 0 and i == ' ':
            continue
        if i == ' ':
            k = 0
            res += 1
        else:
            k = 1
    return res

#441.排列硬币
def arrangeCoins(self, n: int) -> int:
    if n == 1:
        return 1
    base = 1
    res = n
    while True:
        if res < base:
            return base - 1
        elif res == base:
            return base
        else:
            res = res - base
            base += 1
    return base

#直接求等差数列，结果退位保留
def arrangeCoins(self, n: int) -> int:
    return int((0.25 + 2 * n) ** 0.5 - 0.5)

#443.压缩字符串(未掌握)
#双指针
def compress(self, chars: List[str]) -> int:
    if not chars:
        return 0
    index = 0#新字符的下标，由于压缩后的字符串长度一定比之前的短，所以可以使用新的下表然后在原字串上更新
    lens = len(chars)
    i  = 0 #首字母指针
    while i < lens:
        j = i + 1
        while j < lens and chars[j] == chars[i]:
            j += 1
        if j - i > 1:#相同的字符长度大于1，进行压缩：字符+数字
            chars[index] = chars[i]
            index += 1
            strs = str(j - i)
            for s in strs:
                chars[index] = s
                index += 1
        else:#相同字符串的长度等于1，直接写字符，后面不加数字
            chars[index] = chars[i]
            index += 1
        i = j
    return index
#三指针
def compress2(self, chars: List[str]) -> int:
    a,b,c = 0,0,0
    chars.append(1)
    while c < len(chars):
        if chars[b] == chars[c]:
            c += 1
        else:
            chars[a] = chars[b]
            a += 1
            if c - b > 1:
                for i in str(c-b):
                    chars[a] = i
                    a += 1
            b = c
            c += 1
    chars.pop()
    return a

#447.回旋镖的数量
#时间最优
def numberOfBoomerangs(self, points: List[List[int]]) -> int:
    def distance(x1, y1):
        d = collections.Counter((x2 - x1)**2 + (y2 - y1)**2 for x2, y2 in points)
        return sum(t*t - t for t in d.values())#？
    return sum(distance(x1, y1) for x1,y1 in points)
#哈希
def numberOfBoomerangs(self, points: List[List[int]]) -> int:
    res = 0
    points_num = len(points)
    for i in range(points_num):
        hashmap = {}
        for j in range(points_num):
            if i == j:
                continue
            num = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
            hashmap[num] = hashmap[num] + 1 if num in hashmap else 1
        for key in hashmap:
            res += hashmap[key] * (hashmap[key] - 1)

#448.找到所有数组中消失的数字
def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
    s = set(nums)
    return [i for i in range(1, len(nums) + 1) if i not in s]

#453.最小移动次数使数组元素相等
#n-1个元素+1可等价于1个元素-1
def minMoves(self, nums: List[int]) -> int:
    if not nums:
        return 0
    min_num = min(nums)
    res = 0
    for i in nums:
        res += i - min_num
    return res
#455.分发饼干
#贪心算法+双指针遍历
def findContentChildren(self, g: List[int], s: List[int]) -> int:
    g = sorted(g)
    s = sorted(s)
    gi = si = 0
    while gi < len(g) and si < len(s):
        if s[si] >= g[gi]:
            gi += 1
        si += 1
    return gi

#459.重复的子字符串
#假设s可由子串x重复n次构成，s = nx
#s + s = 2nx
#移除s + s开头和结尾的字符，变为(s + s)[1 : -1],则破坏了开头和结尾的子串x
#此时只剩2n-2个子串
#若s在(s + s)[1: -1]中，则有2n-2>=n,n>=2
#即s至少可由x重复两次构成
def repeatedSubstringPattern(self, s: str) -> bool:
    return s in (s + s)[1 : -1]

#463.岛屿的周长
def islandPerimeter(self, grid: List[List[int]]) -> int:
    length = len(grid)
    width = len(grid[0])
    prm = 0
    for i in range(length):
        for j in range(width):
            if grid[i][j] == 1:
                if j == 0 or grid[i][j - 1] == 0:#左边的边
                    prm += 1
                if i == 0 or grid[i - 1][j] == 0:#上边的边
                    prm += 1
    return prm * 2

def hammingDistance(self, x: int, y: int) -> int:
    return bin(x^y).count('1')

#475.供暖器
#重点
#滑动查找
def findRadius(self, houses: List[int], heaters: List[int]) -> int:
    houses.sort()
    heaters.append(float("inf"))
    heaters.sort()
    #逐步往前逼近
    max_dist = 0
    index = 0
    for house in houses:
        while (house >= heaters[index]):#当house大于heater时向右侧移动
            index += 1
        if index > 0:#house 夹在index-1和index之间
            curr_dist = min(heaters[index] - house, house - heaters[index - 1])
        else:#index-1不合法，只需要比较一个值
            curr_dist = abs(heaters[index] - house)
        max_dist = max(max_dist, curr_dist)
    return max_dist

#476.数字的补数
def findComplement(self, num: int) -> int:
    return 2**(len(bin(num))-2)-num-1 #二进制数和它的补数相加和为2**n - 1

#482.密匙格式化
def licenseKeyFormatting(self, S: str, K: int) -> str:
    #逆序
    s = S.upper().replace('-', '')[::-1]
    res = ''
    for i in range(len(s)):
        if i % K == 0 and i!=0:
            res = '-' + res
        res = s[i] + res
    return res

#b保存余数，c为除开头外的组数
def licenseKeyFormatting(self, S: str, K: int) -> str:
    st = S.replace('-', '').upper()
    a = len(st)
    if a == 0:
        return ''
    b = a%K
    if b == 0:
        c = a // K - 1
        b = K
    else:
        c = a // K
    s = st[:b]
    while c != 0:
        s += '-' + st[b:b+K]
        b = b + K
        c -= 1
    return s

#485.最大连续1的个数
def findMaxConsecutiveOnes(self, nums):
    max_num = 0
    res = 0
    for i in nums:
        if i == 1:
            res += 1
            if res > max_num:
                max_num = res
        else:
            res = 0
    return max_num

#492.构造矩形
def constructRectangle(self, area: int) -> List[int]:
    l_w = []
    mid = int(area ** 0.5)
    for i in range(mid, 0, -1):
        if area % i == 0:
            l_w.append(area // i)
            l_w.append(i)
            return l_w

def constructRectangle2(self, area: int) -> List[int]:
    W = int(math.sqrt(area))
    while area % W != 0:
        W -= 1
    return [area // W, W]

#496.下一个更大元素 1
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    if not nums2 or not nums1:
        return []
    ret = []
    maxnum = max(nums2)
    for each1 in nums1:
        flag = 0
        if each1 == maxnum:
            ret.append(-1)
            flag = 1
        for each2index in range(nums2.index(each1)+1, len(nums2)):
            if nums2[each2index] > each1:
                ret.append(nums2[each2index])
                flag = 1
                break
        if flag == 0:
            ret.append(-1)
    return ret


if __name__ == '__main__':
    #13
    #result = romanToInt('MCMXCIV')
    #print(result)
    #14
    #longestCommonPrefix2(None,["flower","flow","flight"])
    #20
    # result = isValid(None,'([)]')
    # print(result)
    #21
    # l = LinkList()
    # l1 = l.initList([1,2,3])
    # l2 = l.initList([1,2,4])
    # mergeTwoLists(None,l1,l2)
    #26
    # nums = [1,1,2]
    # lens = removeDuplicates2(None,nums)
    #23
    # nums = [0,1,2,2,3,0,4,2]
    # val = 2
    # removeElement(None,nums,val)
    #28
    # haystack = "hello"
    # needle = "ll"
    # location = strStr2(None,haystack,needle)
    # print(location)
    #35
    # nums = [1, 3, 5, 6]
    # target = 2
    # result = searchInsert(None, nums, target)
    # print(result)
    #38
    # result = countAndSay(None, 4)
    # print(result)
    #58
    # length = lengthOfLastWord2(None, ' hello  ')
    # print(length)
    #66
    # result = plusOne(None, [1, 2, 3, 9])
    # print(result)
    #67
    #result = addBinary(None,'1010','1011')
    #69
    # ans = mySqrt(None, 4)
    # print(ans)
    #70
    # result = climbStairs(None, 3)
    # print(result)
    # result = climbStairs2(None, 3)
    # print(result)
    #83
    # linklist = LinkList()
    # l1 = linklist.initList([1, 1, 2])
    # result = deleteDuplicates(None, l1)
    # linklist.printList(result)
    #88
    # result = merge(None, [1,2,5], [3, 5, 7, 8, 9])
    # merge(None,[1,2,5,0,0,0,0,0],3,[3,5,7,8,9],5)
    # generate(None, 3)
    #119
    # result = getRow(None, 0)
    # print(result)
    #172
    #result = trailingZeroes(None, 7)
    # print(result)
    #189
    # result = rotate(None, [1,2,3,4,5,6,7], 3)
    # print(result)
    # result = compress(None, ["a","a","b","b","c","c","c"])
    # print(result)
    # result = findRadius(None,[1,2,3],[2])
    #492
    res = constructRectangle(None, 16)
    print(res)