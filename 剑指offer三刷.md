#  [剑指 Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/) 



**一次遍历**

数组长度为n，并且数组内所有元素都在 0~n-1之内，因此若元素不重复，则排序后有`nums[i] = i`，一种方法是先排序，然后遍历数组，遇到重复元素时返回，时间复杂度为`O(logn + n)`

也可以采用一次遍历的方法，不断将元素移到指定位置，即令`nums[i] = i`

```c++
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        for(int i = 0; i < nums.size(); i++){
            while(nums[i] != i){
                if(nums[nums[i]] == nums[i]) return nums[i];
                swap(nums[i],nums[nums[i]]);
            }
        }
        return -1;
    }
};
```



#  [剑指 Offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

```c++
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if(matrix.empty() || matrix[0].empty()) return false;
        //行
        int m = matrix.size();
        //列
        int n = matrix[0].size();
        //从左下角开始查询，向右递增，向上递减
        int i = m - 1, j = 0;
        while(i >= 0 && j < n)
        {
            if(matrix[i][j] > target)
                i--;
            else if(matrix[i][j] < target)
                j++;
            else
                return true;
        }
        return false;
    }
};
```



# [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

## 1. 空间复杂度O(n)解法

```c++
class Solution {
public:
    string replaceSpace(string s) {
        string res;
        for(auto ch : s)
        {
            if(ch != ' ')
                res += ch;
            else
                res +="%20";
        }
        return res;
    }
};
```

## 2. 原地操作解法

```c++
class Solution {
public:
    string replaceSpace(string s) {
        int num = 0;
        for(auto ch : s){
            if(ch == ' ')
                num++;
        }
        int oldSize = s.size();
        s.resize(s.size() + 2 * num);

        for(int i = oldSize - 1, j = s.size() - 1; i >= 0; i--, j--)
        {
            if(s[i] != ' ')
                s[j] = s[i];
                else{
                s[j] = '0';
                s[j-1] = '2';
                s[j-2] = '%';
                j -= 2;
            }
        }
        return s;
    }
};
```



# [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

## 1. 暴力

**按顺序打印，然后反转结果进行输出**

```c++
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        vector<int> res;
        if(!head) return res;
        ListNode* cur = head;
        while(cur)
        {
            res.push_back(cur->val);
            cur = cur->next;
        }

        reverse(res.begin(),res.end());
        return res;
    }
};
```

## 2. 递归

```c++
class Solution {
public:
    void dfs(ListNode* head, vector<int>& res)
    {
        if(!head)
        {
            return;
        }
        dfs(head->next,res); 
        res.push_back(head->val);
    }
    vector<int> reversePrint(ListNode* head) {
        if(!head) return vector<int>{};
        vector<int> res;
        dfs(head,res);
        return res;
    }
};
```

## 3. 栈

```c++
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        stack<int> s;
        while(head){
            s.push(head->val);
            head = head->next;
        }
        vector<int> res;
        while(!s.empty())
        {
            res.push_back(s.top());
            s.pop();
        }
        return res;
    }
};
```



# [剑指 Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

- 前序遍历---根左右

- 中序遍历---左根右

- 前序遍历的首元素为根的val，然后根据根的val去中序遍历中找到左右子树的元素，然后递归构建

  

```c++
class Solution {
public:
/*
    前序遍历---根左右
    中序遍历---左根右
    前序遍历的首元素为根的val，然后根据根的val去中序遍历中找到左右子树的元素，然后递归构建
*/
    unordered_map<int,int> hash;

    TreeNode* buildDfs(vector<int>& preorder, vector<int>& inorder, int pl, int pr, int il, int ir)
    {
        //递归终止条件
        if(pl > pr || il > ir) return nullptr;
        //根据前序遍历首元素构造根节点
        TreeNode* root = new TreeNode(preorder[pl]);
        //寻找根节点元素在中序遍历中的下标
        int index = hash[preorder[pl]];
		// 左子树元素个数为 index - il
        root->left = buildDfs(preorder, inorder, pl + 1, pl + 1 + index - il - 1, il, index - 1);
        root->right = buildDfs(preorder, inorder, pl + 1 + index - il - 1 + 1, pr , index + 1, ir);

        return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        if(preorder.empty()) return nullptr;
        for(int i = 0; i <inorder.size();i++)
            hash[inorder[i]] = i;

        return buildDfs(preorder,inorder,0,preorder.size()-1, 0, inorder.size()-1);
    }
};
```

## 扩展：[106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```c++
class Solution {
    int post_idx;
    unordered_map<int, int> idx_map;
public:
    TreeNode* helper(int in_left, int in_right, vector<int>& inorder, vector<int>& postorder){
        // 如果这里没有节点构造二叉树了，就结束
        if (in_left > in_right) {
            return nullptr;
        }

        // 选择 post_idx 位置的元素作为当前子树根节点
        int root_val = postorder[post_idx];
        TreeNode* root = new TreeNode(root_val);

        // 根据 root 所在位置分成左右两棵子树
        int index = idx_map[root_val];

        // 下标减一
        post_idx--;
        // 构造右子树
        root->right = helper(index + 1, in_right, inorder, postorder);
        // 构造左子树
        root->left = helper(in_left, index - 1, inorder, postorder);
        return root;
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        // 从后序遍历的最后一个元素开始
        post_idx = (int)postorder.size() - 1;

        // 建立（元素，下标）键值对的哈希表
        int idx = 0;
        for (auto& val : inorder) {
            idx_map[val] = idx++;
        }
        return helper(0, (int)inorder.size() - 1, inorder, postorder);
    }
};
```



# [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

1. 定义两个栈s1,s2 
2. 元素存储在s1 中，当要删除元素时，应当删除s1 的**栈底元素**，因此将s1中的元素依次出栈并压入s2中，此时s2的栈顶元素即为s1**栈底元素**，将其出栈
3. 然后将s2元素依次再返回s1中

```c++
class CQueue {
public:
    CQueue() {

    }
    
    void appendTail(int value) {
       s1.push(value); 
    }
    
    int deleteHead() {
        if(s1.empty()) return -1;
        while(!s1.empty()){
            s2.push(s1.top());
            s1.pop();
        }
        int res = s2.top();
        s2.pop();
        while(!s2.empty())
        {
            s1.push(s2.top());
            s2.pop();
        }
        return res;
    }
private:
    stack<int> s1;
    stack<int> s2;
};
```

**步骤三很多余。将元素全部压入s2后，当有删除操作时，继续删除s2中元素即可，直到s2为空，输入元素还是直接往s1压入，当s2为空时，再将s1元素压入其中，进行删除**

```c++
class CQueue {
public:
    CQueue() {

    }
    
    void appendTail(int value) {
        s1.push(value);
    }
    
    int deleteHead() {
        if(s2.empty())
        {
            if(s1.empty())  return -1;
            while(!s1.empty()){
                s2.push(s1.top());
                s1.pop();
            }
            int res = s2.top();
            s2.pop();
            return res;
        }else{
            int res = s2.top();
            s2.pop();
            return res;
        }
        return -1;
    }
private:
    stack<int> s1;
    stack<int> s2;
};
```



# [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

**简单动态规划**

```c++
class Solution {
public:
    int fib(int n) {
        if(n < 2) return n;
        vector<int> dp(n+1,0);
        dp[1] = 1;
        for(int i = 2; i <= n; i++)
        {
            dp[i] = (dp[i-1] + dp[i-2]) % 1000000007;
        }
        return dp[n];
    }
};
```



# [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

**简单动态规划**

```c++
class Solution {
public:
    int numWays(int n) {
        if(n < 2) return 1;
        vector<int> dp(n+1,1);
        for(int i = 2; i <= n; i++)
        {
            dp[i] = (dp[i-1] + dp[i-2]) % 1000000007;
        }
        return dp[n];
    }
};
```



# [剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

## 暴力：

```c++
class Solution {
public:
    int minArray(vector<int>& numbers) {
        int res = INT_MAX;
        for(auto num : numbers){
            res = min(res,num);
        }
        return res;
    }
};
```

## 二分法：

- 数组旋转以后分为了两部分，两部分均为升序序列

- 右边的序列中所有值均小于左边序列的最小值，即num[0]

  - 若当前值大于num[0]，说明此时处于左边序列，`left = mid + 1`
  - 若当前值小于num[0]，但是大于其左值，则`right = mid`

  ```c++
  
  class Solution {
  public:
      int minArray(vector<int>& numbers) {
          int n = numbers.size() - 1;
          if(n < 0) return -1;
          //去除尾部与首部相同的元素
          while(n > 0 && numbers[0] == numbers[n]) n--;
          if(numbers[n] > numbers[0]) return numbers[0];
          int l = 0,r = n;
          while(l < r)
          {
              int mid = l + r >> 1;
              if(numbers[mid] < numbers[0])   r = mid;
              else l = mid + 1;
          }
          return numbers[r];
      }
  };
  ```

# [剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

