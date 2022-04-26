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

**重刷此题，第一想法是利用回溯算法，当在矩阵中找到与word开头字符相同的字符时，递归遍历矩阵中该字符的上下左右四个方向，寻找word中的下一个字符**

**代码如下：**

```c++
class Solution {
public:
    bool res = false;
    string path;
    void backTracking(vector<vector<char>>& board, const string& word, int startIndex, int x, int y)
    {
        if(path == word){
            res = true;
            return;
        }
        if(x >= board.size()|| y >= board[0].size()) return;
        int dx[] = {-1, 0, 1, 0};
        int dy[] = {0, 1, 0, -1};

        if(word[startIndex] == board[x][y])
        {
            path.push_back(board[x][y]);
            for(int i = 0; i < 4; i++){
                backTracking(board,word,startIndex + 1, x + dx[i], y + dy[i]);
            }
        }
        

    }
    bool exist(vector<vector<char>>& board, string word) {
        for(int i = 0; i < board.size(); i++){
            for(int j = 0; j < board[0].size(); j++){
                if(board[i][j] == word[0]){
                    backTracking(board,word,0,i,j);
                }
            }
        }
        return res;
    }
};
```

**此时，当给定输入分别为`[["a","a"]] "aaa"`时报错，手动推算了一遍，发现是遍历到矩阵第二个a时，此时再向左遍历，重复计算了矩阵的第一个字符，因此，每遍历一个字符，应该将其标记，防止重复遍历**

**代码如下：**

```c++
class Solution {
public:
    bool res = false;
    string path;
    void backTracking(vector<vector<char>>& board, const string& word, int startIndex, int x, int y)
    {
        if(path == word){
            res = true;
            return;
        }
        if(x >= board.size() || y >= board[0].size()) return;

        int dx[] = {-1, 0, 1, 0};
        int dy[] = {0, 1, 0, -1};

        if(word[startIndex] == board[x][y])
        {
            path.push_back(board[x][y]);
            char temp = board[x][y];
            board[x][y] = '*';
            for(int i = 0; i < 4; i++){
                int a = x + dx[i];
                int b = y + dy[i];
                backTracking(board,word,startIndex + 1, a, b);
            }
            path.pop_back();
            board[x][y] = temp;
        }
    }
    
    bool exist(vector<vector<char>>& board, string word) {
        for(int i = 0; i < board.size(); i++){
            for(int j = 0; j < board[0].size(); j++){
                if(board[i][j] == word[0]){
                    backTracking(board,word,0,i,j);
                }
            }
        }
        return res;
    }
};
```

**喜闻乐见，又报错了，原因是超出时间限制**

**更改代码如下：**

```c++
class Solution {
public:
    bool backTracking(vector<vector<char>>& board, const string& word, int startIndex, int x, int y)
    {
        if(board[x][y] != word[startIndex]) return false;
        //已经遍历到了word的最后一个字符，则直接返回true
        if(startIndex == word.size() - 1){
            return true;
        }

        int dx[] = {-1, 0, 1, 0};
        int dy[] = {0, 1, 0, -1};

        char temp = board[x][y];
        board[x][y] = '*';
        for(int i = 0; i < 4; i++){
            int a = x + dx[i];
            int b = y + dy[i];
            if(a >= 0 && a < board.size() && b >= 0 && b < board[a].size()){
                if(backTracking(board,word,startIndex + 1, a, b)) return true;
            }
        }
        board[x][y] = temp;
        return false;

    }
    bool exist(vector<vector<char>>& board, string word) {
        for(int i = 0; i < board.size(); i++){
            for(int j = 0; j < board[0].size(); j++){
                if(backTracking(board,word,0,i,j))
                    return true;
            }
        }
        return false;
    }
};
```



# [剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

```c++
class Solution {
public:
    int get_sum(pair<int, int> p) {
        int s = 0;
        while (p.first) {
            s += p.first % 10;
            p.first /= 10;
        }
        while (p.second) {
            s += p.second % 10;
            p.second /= 10;
        }
        return s;
    }

    int movingCount(int m, int n, int k) {
        int res = 0;
        if(!m || !n) return 0;
        vector<vector<bool>> st(m,vector<bool>(n));
        queue<pair<int,int>> q;

        q.push({0,0});

        int dx[4] = {-1,0,1,0} ,dy[4] = {0,1,0,-1};

        while(!q.empty()){
            auto t = q.front();
            q.pop();

            if(get_sum(t) > k || st[t.first][t.second]) continue;
            res++;
            st[t.first][t.second] = true;
            for(int i = 0;i < 4;i++){
                int x = t.first + dx[i], y = t.second + dy[i];
                if(x >= 0 && x < m && y >= 0 && y < n){
                    q.push({x,y});
                }
            }
        }
        return res;
    }
};

```



# [剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

## 1. **贪心：每次尽可能去切割出3**

```c++
class Solution {
public:
    int cuttingRope(int n) {
        if(n < 4) return n - 1;
        if(n == 4) return 4;
        int res = 1;
        while(n > 4)
        {
            n -= 3;
            res *= 3;
        } 

        return res * n;
    }
};
```

## 2. 动态规划

- dp数组定义：绳子长度为 i 时，可以剪出的最大乘积为dp[i]
- 状态转移：
  - 从2到绳子长度 i 遍历，剪下一段长度 j 后，可以选择继续剪，也可以选择不再剪
  - `dp[i] = max(j * (i - j), j * dp[i-j])`
  - 遍历过程中不断取最大值，`dp[i] = max(max(j * (i - j), j * dp[i-j]), dp[i])`



**代码如下：**

```c++
class Solution {
public:
    int cuttingRope(int n) {
        vector<int> dp(n+1);
        dp[2] = 1;
        for(int i = 3; i <= n; i++)
        {
            for(int j = 2; j < i; j++)
            {
                dp[i] = max(j*dp[i-j],max(dp[i],j * (i-j)));

            }
        }
        return dp[n];
    }
};
```



# [剑指 Offer 14- II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

**该题要注意整形溢出问题,与上题相比，用长整型long代替了整形int**

```c++
class Solution {
public:
    int cuttingRope(int n) {
        if(n < 4) return n - 1;
        if(n == 4) return 4;
        long res = 1;
        while(n > 4)
        {
            n -= 3;
            res = res * 3 % 1000000007;
        } 

        return res * n % 1000000007;
    }
};
```



# [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

**简单位运算**

```c++
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int res = 0;
        for(int i = 0; i < 32; i++)
        {
            if((n >> i) & 1) res++;
        }
        return res;
    }
};
```



# [剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

**快速幂：**

![IMG_20211205_151144](F:\A3-git_repos\数据结构与算法_notes\图片\IMG_20211205_151144.jpg)

## **快速幂模板：**

```c++
//非递归快速幂
int qpow(int a, int n){
    int ans = 1;
    while(n){
        if(n&1)        //如果n的当前末位为1
            ans *= a;  //ans乘上当前的a
        a *= a;        //a自乘
        n >>= 1;       //n往右移一位
    }
    return ans;
}
```

## 解法：

```c++
class Solution {
public:
    double myPow(double x, int n) {
        double res = 1;
        if(x == 1) return x;
        bool flag = n > 0 ? true : false;
        long n1 = abs(n);
        while(n1)
        {
            if(n1 & 1) res *= x;
            x *= x;
            n1 >>= 1;
        }
        return flag ? res : 1/res;
    }
};
```

**该题要特别注意，当 n 为INT_MIN时，普通整型无法存储，必须用long型存储**

# [剑指 Offer 17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

直接暴力输出

```c++
class Solution {
public:
    vector<int> printNumbers(int n) {
        vector<int> res;
        int i  =  1;
        while(n--){
            i *= 10;
        }
        for(int j = 1; j < i; j++){
            res.push_back(j);
        }
        return res;
    }
};
```

**考虑大数越界时，用字符数组存储**

```c++
class Solution {
private:
    vector<int> nums;
    string s;
public:
    vector<int> printNumbers(int n) {
        s.resize(n, '0');
        dfs(n, 0);
        return nums;
    }
    
    // 枚举所有情况
    void dfs(int end, int index) {
        if (index == end) {
            save(); return;
        }
        for (int i = 0; i <= 9; ++i) {
            s[index] = i + '0';
            dfs(end, index + 1);
        }
    }
    
    // 去除首部0
    void save() {
        int ptr = 0;
        while (ptr < s.size() && s[ptr] == '0') ptr++;
        if (ptr != s.size())
            nums.emplace_back(stoi(s.substr(ptr)));
    }
};
```



# [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

```c++
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        if(!head) return head;
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;
        ListNode *prev = dummy;
        ListNode* cur = head;
        while(cur)
        {
            if(cur->val != val)
            {
                prev = cur;
                cur = cur->next;
            }else{
                prev->next = cur->next;
                return dummy->next;
            }

        }
        return dummy->next;
    }
};
```

# [剑指 Offer 19. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

## 1. 动态规划数组定义以及状态定义

- `dp[i][j]: 字符串s的前i个字符与模式串p的前j个字符的匹配情况`
- 若 `s[i-1] == p[j-1]`
  - `dp[i][j] = dp[i-1][j-1]`
- 若`s[i-1] != p[j-1]`
  - 若`p[j-1] == '.'`
    - `dp[i][j] == dp[i-1][j-1] `，该类情况可以合并到第一类中
  - 若`p[j-1] == '*'`
    - 则可以匹配0个`p[j-2]`或者匹配 1 个
      - 匹配 0 个时：`dp[i][j] = dp[i][j-2]`
      - 匹配 1 个时：`dp[i][j] = dp[i-1][j] && (s[i-1] == p[j-2] || p[j-2] == '.')`
  - 若`p[j-1]为小写字母`
    - 则`dp[i][j] = false`

**综上，一共可以分为两种情况，分别为`p[j-1] == '*' 或者 p[j-1] != '*'`**

## 2. DP数组初始化

- 当给定字符串以及模式串均为空是，两者当然是匹配的(空字符匹配空字符)

  - `dp[0][0] = true`

- 当模式字符串为空，而给定字符串不为空时，结果当然是false

  - `for(int i = 1; i <= ns; i++) dp[i][0] = false;`

- 当给定字符串为空，而模式字符串不为空时，只有当模式字符串的偶数位上为`*`时，两者才匹配

  - ```c++
     for(int i = 2; i <= np; i += 2){
                if(p[i - 1] == '*') dp[0][i] = dp[0][i-2];
            }
    ```



## 3. 代码

```c++
class Solution {
public:
    bool isMatch(string s, string p) {
        int ns = s.size();
        int np = p.size();
        vector<vector<bool>> dp(ns+1,vector<bool>(np+1,false));
        dp[0][0] = true;
        for(int i = 1; i <= ns; i++) dp[i][0] = false;
        for(int i = 2; i <= np; i += 2){
            if(p[i - 1] == '*') dp[0][i] = dp[0][i-2];
        }

        for(int i = 1; i <= ns; i++)
        {
            for(int j = 1; j <= np; j++)
            {
                if(p[j-1] == '*'){
                    dp[i][j] = dp[i][j-2] || (dp[i-1][j] && (s[i-1] == p[j-2] || p[j-2] == '.'));
                }else{
                    if(s[i-1] == p[j-1] || p[j-1] == '.')
                        dp[i][j] = dp[i-1][j-1];
                    else
                        dp[i][j] = false;
                }
            }
        }

        return dp[ns][np];
    }
};
```

# [剑指 Offer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

**牛逼啊这题，提交了17次，错了16次，一步一步根据错误信息完善代码，终于过了！虽然菜，但好歹是自己一步一步写的**

## 1. 思路：

根据题意，初步思路如下：

- 首先去除字符串前后空格，以及首位的正负号，**特别注意，去除空格以后要判断字符串为否为空，这是我出错的一个点**

  - ```c++
    //去除前后空格
    while(s[i] == ' ') i++;
    while(s[j] == ' ') j--;
    
    if(i >= s.size() || j < 0) return false;
    
    int dot = 0, eNum = 0, flag = 0;
    //去除首位正负号
    if(s[i] == '+' || s[i] == '-'){
        i++;
        flag++;
    }
    ```

    

- 然后遍历去除前后空格之后的字符串，记录正负号数量、`e E`的数量以及点的数量

  - 在字符串中遇到非`e E`的英文字符，直接返回false
  - 遇到正负号以后
    - 因为已经提前处理过首位的正负号，因此第二个正负号只能出现在 e E 的后面，判断其前一位是否为 e E，若不是，直接返回false，若是，则continue
  - 遇到 点
    - 如果之前已经有点了，则直接返回false，一个字符串中只能有一个点
    - 如果点在字符串首部并且字符串长度不为1，且点后紧跟着一个整数，合法
    - 如果点在字符串中部，且前一位是整数，合法
    - 否则，不合法，返回false
  - 遇到 e E
    - 整个字符串只能有一个 e E，否则不合法
    - 字符串长度为 1 ,不合法
    - e前面为正负号，不合法
    - e后面一位可以为正负号，其他位只能为整数

**整体看来，其实该题就分为两部分，e之前的字符为一部分，e之后的字符为一部分**

- e之前的字符必须为小数或者整数，**注意`3.e100`也是合法的,即一个整数带一个点是一个合法的数。形如`1.`， `2.`, ` 3.`,均是合法的数**
- e之后的字符必须为整数，e之后一位可以是正负号，之后各位必须为`0 ~ 9`的整数

**对我而言，对于出现点的情况的判断非常麻烦，这也是我一直出错的原因，然后就是一些细节处理的不够到位，比如去除首位空格以后判断字符串是否为空、字符串中部有空格,这都是我没有考虑完整的地方**

**综上，下次碰上这题我感觉我还是要出错**

```c++
class Solution {
public:
    bool isNumber(string s) {
        if(s.empty()) return false;
        int i = 0, j = s.size() - 1;
        //去除前后空格
        while(s[i] == ' ' && i <= j) i++;
        while(s[j] == ' ' && j >= i) j--;

        if(i >= s.size() || j < 0) return false;
        
        int dot = 0, eNum = 0, flag = 0;
        //去除首位正负号
        if(s[i] == '+' || s[i] == '-'){
            i++;
            flag++;
        }

        for(int k = i; k <= j; k++)
        {
            if(s[k] == ' ') return false;
            if(s[k] >= 'a' && s[k] <= 'z' && s[k] != 'e' || s[k] >= 'A' && s[k] <= 'Z' && s[k] != 'E') return false;
            if(s[k] == '+' || s[k] == '-'){
                if(k == i) return false;
                if(k > i && (s[k-1] != 'e' || s[k-1] != 'E') ) return false;
                else{
                    continue;
                }
            }

            if(s[k] == '.'){
                if(dot > 0) return false;
                if(k > i && (s[k-1] >= '0' && s[k-1] <= '9')){
                    dot++;
                    continue;
                }
                if(k == i && k < j && (s[k+1] >= '0' && s[k+1] <= '9')){
                    dot++;
                    continue;
                }
                return false;
            }
            
            if(s[k] == 'e' || s[k] == 'E'){
                if(eNum > 0) return false;
                if(k == i || k == j) return false;
                if(s[k-1] == '+' || s[k-1] == '-') return false;
                for(int x = k + 1; x <= j; x++){
                    if(x == k + 1 && (s[x] == '+' || s[x] == '-') && x != j) continue;
                    if(s[x] < '0' || s[x] > '9') return false;
                }
                return true;
            }

        }
        return true;
    }
};
```



## 2. 改善：

**我只是用了两个指针指向首尾，其实处理完首尾后，完全可以利用substr将字符串截出来**

贴上别人的代码

```c++
class Solution {
public:
    bool isNumber(string s) {
        //首先删除前后空格
        int i = 0, j = s.size()-1;
        while(i <= j && s[i] == ' ') i++;
        while(i <= j && s[j] == ' ') j--; 
        if(i > j) return false;
        
        s = s.substr(i,j - i + 1);
        if(s[0] == '+' || s[0] == '-') s = s.substr(1);     //忽略正负号
        if(s.empty() || (s[0] == '.' && s.size() == 1)) return false;  

        int dot = 0,e = 0;
        for(int i = 0; i < s.size();i++){
            if(s[i] >= '0' && s[i] <= '9') ;
            else if( s[i] == '.'){
                dot++;
                if(dot > 1 || e) return false;  // '.'之前不可出现'.'和'e'
            }
            else if(s[i] == 'e' || s[i] == 'E'){
                e++;
                 //整串中可能出现一个'e'，且不能为首字符，且其后不能为空，'e'之前为'.'且'.'前面为空
                if(i == 0 || i+1 == s.size() || e > 1 || s[i-1] == '.' && i == 1) return false;   
                if(s[i+1] == '+' || s[i + 1] == '-'){             //若'e'后为正负号，则正负号后不能为空
                    if(i+2 == s.size()) return false;
                    i++;
                }
            }
            else return false;
        }
        return true;
    }
};
```



# [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

**简单双指针**

```c++
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        if(nums.empty()) return nums;
        int i = 0, j = nums.size() - 1;
        while(i < j)
        {
            while(nums[i] % 2 == 0 && j > i){
                swap(nums[i],nums[j--]);
            }
            i++;
            while(nums[j] % 2 == 1 && i < j){
                swap(nums[j],nums[i++]);
            }
            j--;
        }
        return nums;
    }
};
```



# [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

 **简单双指针**

```c++
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        if(!head) return head;
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;
        ListNode* fast = dummy;
        ListNode* slow = dummy;
        while(fast && k--){
            fast = fast->next;
        }
        while(fast->next){
            slow = slow->next;
            fast = fast->next;
        }
        return slow->next;
    }
};
```



# [剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

**纯属于自己把自己往坑里带，习惯性的加了一个哑节点，结果delete的时候老报错，于是强行回顾了一下指针相关的知识**

- 当函数输入一个空指针时，结束后`prev == dummy`,此时`delete dummy`后，`prev`指向了一块非法内存，于是报错
- 因此加了一个判断，当反转结束后，若`prev == dummy`,则先令`prev = nullptr`，再`delete dummy`

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;
        ListNode* prev = nullptr;
        ListNode* cur = dummy;
        while(cur)
        {
            ListNode* next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        
        if(head) head->next = nullptr;
        if(prev == dummy) prev = nullptr;
        delete dummy;
        dummy = nullptr;
        return prev;
    }
};
```



# [剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

**简单双指针**

```c++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(-1);
        ListNode* cur = dummy;
        ListNode* head1 = l1;
        ListNode* head2 = l2;
        while(head1 && head2)
        {
            if(head1->val < head2->val){
                cur->next = head1;
                head1 = head1->next;
            }else{
                cur->next = head2;

            }
            cur = cur->next;
        }
        if(head1) cur->next = head1;
        if(head2) cur->next = head2;
        
        return dummy->next;
    }
};
```



# [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

## 1. 错解

```c++
class Solution {
public:
    bool dfs(TreeNode* A, TreeNode* B)
    {
        if(!B) return true;
        if(!A && B) return false;

        if(A->val == B->val) return dfs(A->left,B->left) && dfs(A->right,B->right);
        return dfs(A->left,B) || dfs(A->right,B);
    }
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(!A || !B) return false;
        return dfs(A,B);
    }
};
```

## 2. 正解

```c++
class Solution {
public:
    bool dfs(TreeNode* A, TreeNode* B)
    {
        if(!B) return true;
        if(!A && B) return false;

        if(A->val == B->val) return dfs(A->left,B->left) && dfs(A->right,B->right);
        return false;
    }
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(!B || !A) return false;
        return dfs(A,B) || isSubStructure(A->left,B) || isSubStructure(A->right,B);
    }
};
```

## 3. 思考

**该题分为两步：**

1. 在树A中找到与树B根节点相同值的节点X
2. 然后判断树A中以节点X为根节点的子树是否包含树B

**两者的递归终止条件是不同的：**

**对于步骤1 ，由于规定空树不为任何树的子树，因此只要树A或者树B两者之一为空，就返回false**

**对于步骤2，当遍历到树B的空节点时，不论树A当前节点是否为空，都应返回true,表示树B的左或右子树匹配完成，而当树A当前节点为空而树B当前节点不为空时，显然匹配失败，返回false**

**我的错误也是基于上述原因，没有分清楚两者区别**

# [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

**简单递归**

```c++
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if(!root) return root;

        TreeNode* l = root->left;
        root->left = root->right;
        root->right = l;

        mirrorTree(root->left);
        mirrorTree(root->right);
        
        return root;
    }
};
```



# [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

**简单递归**

```c++
class Solution {
public:
    bool isMirror(TreeNode* root1, TreeNode* root2)
    {
        if(!root1 && !root2) return true;
        if(!root1 && root2) return false;
        if(root1 && !root2) return false;
        if(root1->val == root2->val) return isMirror(root1->left,root2->right) && isMirror(root1->right,root2->left);
        return false;
    }
    bool isSymmetric(TreeNode* root) {
        if(!root) return true;
        return isMirror(root->left,root->right);
    }
};
```



# [剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

**简单模拟题**

自己做的时候忽略了一个小细节，导致错误

## 正解：

**每次打印完一行或者一列以后，要立即判断边界条件是否合法，若不合法，立即终止循环，返回输出结果，第一次提交时由于没有注意到这点，造成了错误**

```c++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> res;
        if(matrix.empty() || matrix[0].empty()) return res;
        int m = matrix.size();
        int n = matrix[0].size();
        
        int high = 0, low = m - 1, left = 0, right = n - 1;
        while(left <= right && high <= low)
        {
            for(int i = left; i <= right; i++)
                res.push_back(matrix[high][i]);
            high++;
            //判断边界条件
            if(high > low) break;

            for(int i = high; i <= low; i++)
                res.push_back(matrix[i][right]);
            right--;
            //判断边界条件
            if(right < left) break;

            for(int i = right; i >= left; i--)
                res.push_back(matrix[low][i]);
            low--;
            //判断边界条件
            if(low < high) break;

            for(int i = low; i >= high; i--)
                res.push_back(matrix[i][left]);
            left++;
            //判断边界条件
            if(left > right) break;
        }
        return res;
    }
};
```

**精简版**

```c++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> res;
        if(matrix.empty() || matrix[0].empty()) return res;
        int high = 0;
        int low = matrix.size()-1;
        int left = 0;
        int right = matrix[0].size()-1;

        while(true)
        {
            for(int i = left; i <= right; i++)  res.push_back(matrix[high][i]);
            if(++high > low) break;
            for(int i = high; i <= low; i++) res.push_back(matrix[i][right]);
            if(--right < left) break;
            for(int i = right; i >= left; i--) res.push_back(matrix[low][i]);
            if(--low < high) break;
            for(int i = low; i >= high; i--) res.push_back(matrix[i][left]);
            if(++left > right) break;
        }
        return res;
    }
};
```



# [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

**可以用一个栈也可以用两个栈，但是两种方法本质上是相同的**

```c++
class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {

    }
    
    void push(int x) {
        if(s.empty()) s.push(pair<int,int>{x,x});
        else{
            pair<int,int> num;
            num.first = x;
            num.second = x > s.top().second ? s.top().second : x;
            s.push(num);
        }
    }
    
    void pop() {
        s.pop();
    }
    
    int top() {
        return s.top().first;
    }
    
    int min() {
        return s.top().second;
    }
private:
    stack<pair<int,int>> s;
};
```



# [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

**简单模拟题**

```c++
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        stack<int> s;
        int i = 0;
        for(auto num : pushed){
            s.push(num);
            while(!s.empty() && s.top() == popped[i]){
                s.pop();
                i++;
            }
        }
        return s.empty();
    }
};
```



# [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

**简单层序遍历**

```c++
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) {
        vector<int> res;
        if(!root) return res;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty())
        {
            TreeNode* cur = q.front();
            res.push_back(cur->val);
            q.pop();
            if(cur->left) q.push(cur->left);
            if(cur->right) q.push(cur->right);
        }
        return res;
    }
};
```



# [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty())
        {
            int size = q.size();
            vector<int> level;
            for(int i = 0; i < size; i++)
            {
                TreeNode* cur = q.front();
                q.pop();
                level.push_back(cur->val);
                if(cur->left) q.push(cur->left);
                if(cur->right) q.push(cur->right);
            }
            res.push_back(level);
        }
        return res;
    }
};
```



# [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        queue<TreeNode*> q;
        q.push(root);
        int flag = 1;
        while(!q.empty())
        {
            int size = q.size();
            vector<int> level;
            for(int i = 0; i < size; i++)
            {
                TreeNode* cur = q.front();
                q.pop();
                level.push_back(cur->val);
                if(cur->left) q.push(cur->left);
                if(cur->right) q.push(cur->right);
            }
            if(flag % 2 == 0) reverse(level.begin(),level.end());
            res.push_back(level);
            flag++;
        }
        return res;
    }
};
```



# [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

**该题给定一棵二叉搜索树，对于二叉搜索树，有　左子树的值　＜　根节点值　＜　右子树的值，而后序遍历序列的顺序是`左 右 根`，因此可以根据以上的大小关系，递归判断左子树序列、右子树序列是否满足以上关系**

- 给定数组的尾部元素即为根节点值root
- 遍历数组，寻找第一个比根节点大的值n，则 n 之前为左子树后序遍历序列，n 之后root 之前即为后序遍历序列
- 上面遍历的过程中已经确定了左子树都是小于root的，因此还要判断右子树序列是否均大于root,若不是，则返回false
- 递归左右子树序列



```c++
class Solution {
public:
    bool dfs(vector<int>& postorder, int i, int j)
    {
        if(i >= j) return true;
        int root = postorder[j];
        int p = i;
        while(postorder[p] < root) p++;
        for(int k = p; k < j; k++)
            if(postorder[k] < root) return false;
        
        return dfs(postorder,p,j-1) && dfs(postorder,i,p-1);
    }
    bool verifyPostorder(vector<int>& postorder) {
        if(postorder.empty()) return true;
        return dfs(postorder,0,postorder.size() - 1);
    }
};
```



# [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

**根据代码随想录的回溯套路组织代码，中间踩了不少坑**

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backTracking(TreeNode* root, int target, int sum)
    {
        if(!root) return;
        path.push_back(root->val);
        sum += root->val;
		//终止条件这里一定要判断当前节点是不是叶子节点，不然也会报错
        if(sum == target && !root->left && !root->right){
            res.push_back(path);
            //这一步必须加上，一开始没加上，发现会少pop一个元素
            path.pop_back();
            return;
        }

        backTracking(root->left,target,sum);
        backTracking(root->right,target,sum);

        path.pop_back();
       //传值方式传入，不需要回溯
       // sum -= root->val;

    }
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        if(!root) return res;
        backTracking(root,target,0);
        return res;
    }
};
```

**自己写个小递归**

```c++
class Solution {
public:
    vector<vector<int>> Path ;
    //递归遍历每一条路径
    void dfs(TreeNode* root,vector<int> path,int targetSum,int cur_sum)
    {
        if(!root)  return;
        cur_sum += root->val;
        path.push_back(root->val);
        if(!root->left && !root->right && cur_sum == targetSum)
        {   
            Path.push_back(path);
        }
        dfs(root->left,path,targetSum ,cur_sum);
        dfs(root->right,path,targetSum ,cur_sum);
        //由于是传值方式传入的path,每一层递归都是独立的，互不影响，因此不需要执行pop_back操作
        //path.pop_back();
    }

    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        if(!root) return Path;
        vector<int> path;
        dfs(root,path,targetSum,0);
        return Path;
    }
};
```



# [剑指 Offer 35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

**哈希表**

```c++
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(!head) return nullptr;
        unordered_map<Node*,Node*> hash;
        Node* cur = head;
        while(cur)
        {
            hash[cur] = new Node(cur->val);
            cur = cur->next;
        }

        cur = head;

        while(cur)
        {
            hash[cur]->next = hash[cur->next];
            hash[cur]->random = hash[cur->random];
            cur = cur->next;
        }

        return hash[head];
    }
};
```



# [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

## 暴力法：

二叉搜索树的中序遍历为升序序列，将此升序序列存入数组，然后一次遍历更改每个节点的左右指针，最后特殊处理首尾两节点的指针，处理结束后返回数组首元素。

```c++
class Solution {
public:
    void dfs(Node* root, vector<Node*>& path)
    {
        if(!root) return;
        dfs(root->left,path);
        path.push_back(root);
        dfs(root->right,path);
    }
    Node* treeToDoublyList(Node* root) {
        if(!root) return root;
        vector<Node*> path;
        dfs(root,path);

        for(int i = 0; i < path.size() - 1; i++)
        {
            path[i]->right = path[i+1];
            path[i+1]->left = path[i];
        }
        path[0]->left = path[path.size() - 1];
        path[path.size() - 1]->right = path[0];

        return path[0];
    }
};
```

## 改进：

**直接在递归过程中改变左右指针指向，递归过程中要记录前驱节点**

```c++
class Solution {
public:
    Node* pre;
    Node* head;
    void dfs(Node* root)
    {
        if(!root) return;
        dfs(root->left);
        //
        if(!pre) head = root;
        else pre->right = root;
        root->left = pre;
        pre = root;
        //
        dfs(root->right);
    }
    Node* treeToDoublyList(Node* root) {
        if(!root) return NULL;
        dfs(root);
        head ->left = pre;
        pre->right = head;
        return head;
    }
};
```



# [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

## 1. 序列化

利用层序遍历，将二叉树中的所有节点都打印出来，**包括叶子节点的左右孩子即空节点**

- 遇到空节点时，用字符`#`表示
- 节点与节点之间用`,`分离

```c++
string serialize(TreeNode* root) {
        string res;
        if(!root) return res;
        queue<TreeNode*> q;
        q.push(root);
        int level = 1;
        while(!q.empty())
        {
            TreeNode* cur = q.front();
            q.pop();
            if(!cur)
            {
                res += '#';
                res += ',';
            }
            else{
                res += to_string(cur->val);
                res += ',';
                q.push(cur->left);
                q.push(cur->right);
            }
        }
        return res;
    }
```

## 2. 反序列化

对于层序遍历序列，**一个节点与其左右孩子节点的位置是固定的**,对于一个二叉树层序遍历结果`[1,2,3,null,null,4,5]`，根节点的左右孩子为`2,3`，而`2`的左右孩子节点为`null,null`,`3`的左右孩子节点为`4,5`

对此可以用一下代码来表示

```c++
int pos = 1;
for(int i = 0; i < v.size(); ++i)
{
    if(v[i] == nullptr)
    continue;
    v[i]->left = v[pos++];
    v[i]->right = v[pos++];
}
```

**首次错误提交*

```c++
  TreeNode* deserialize(string data) {
        if(data.empty()) return nullptr;
        vector<TreeNode*> v;
        for(auto s : data)
        {
            if(s == ',') continue;
            if(s == '#'){
                v.push_back(nullptr);
            }
            else{
                TreeNode* node = new TreeNode(s - '0');
                v.push_back(node);
            }
        }

        int pos = 1;
        for(int i = 0; i < v.size(); ++i)
        {
            if(v[i] == nullptr)
                continue;
            v[i]->left = v[pos++];
            v[i]->right = v[pos++];
        }
        return v[0];
    }
```

**我的这段代码出错出在对字符串的处理，我的想法是对字符串的每一个字符进行提取，这种想法就默认了一个字符代表一个数字，即当出现负数或者两位数时，提取的字符就是错的，因此对该部分要进行修改：因为数字之间用逗号隔开，因此两个逗号之间的所有字符合在一起才是一个完整的数字**

```c++
 TreeNode* deserialize(string s) {
        if(s.empty()) return nullptr;
        vector<TreeNode*> v;
        for(int i = 0; i < s.size(); i++)
        {
            //遇到逗号跳过
            if(s[i] == ',') continue;
            //遇到'#'就加入空指针
            else if(s[i] == '#')
                v.push_back(nullptr);
            else{
                //定义一个字符用来提取数字
                string temp = "";
                int j = i;
                while(s[j] != ','){
                    temp += s[j];
                    j++;
                }
                TreeNode * node = new TreeNode (stoi(temp));
                v.push_back(node);
                i = j;
            }
        }

        int pos = 1;
        for(int i = 0; i < v.size(); ++i)
        {
            if(v[i] == nullptr)
                continue;
            v[i]->left = v[pos++];
            v[i]->right = v[pos++];
        }
        return v[0];
    }
```

**以上的字符提取过程还可以进行修改**

```c++
		int j = 0;
        while(j < data.size())
        {
            string stmp = "";
            while(data[j] != ',')
            {
                stmp += data[j];
                j++;
            }

            if(stmp == "#")
            {
                nodes.push_back(nullptr);
            }
            else
            {
                int tmp = atoi(stmp.c_str());
                TreeNode* newnode = new TreeNode(tmp);
                nodes.push_back(newnode);
            }
            j++;
        }
```



# [剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

**经典回溯全排列**

```c++
class Solution {
public:
    vector<string> res;
    string path;
    void backTracking(string s,vector<bool>& used)
    {
        if(path.size() == s.size())
        {
            res.push_back(path);
            return;
        }
        for(int i = 0; i < s.size(); i++)
        {
            if(i > 0 && s[i] == s[i - 1] && used[i-1] == false) continue;
            if(used[i] == false)
            {
                path.push_back(s[i]);
                used[i] = true;
                backTracking(s,used);
                used[i] = false;
                path.pop_back();
            }
        }
    }
    vector<string> permutation(string s) {
        if(s.empty()) return res;
        vector<bool> used(s.size(),false);
        sort(s.begin(),s.end());
        backTracking(s,used);
        return res;
    }
};
```



# [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        int n = nums.size();
        return nums[n/2];
    }
};
```



# [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

## 1. 暴力法

**排序，然后输出前N个元素**

```c++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        if(arr.empty() || k > arr.size()) return vector<int>{};
        sort(arr.begin(),arr.end());
        vector<int> res(arr.begin(),arr.begin() + k);
        return res;
    }
};
```



## 2. 快排

**快排的思想就是以某数为基准，将数组以该基准数分为两段子数组，从左边子数组找一个比基准数的数，从右边子数组找一个比基准数小的数，然后将两数交换。基于此思想，当基准 左边的元素个数为K时，左边子数组中的元素即为K个最小的数**

```c++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        if (k >= arr.size()) return arr;
        return quickSort(arr, k, 0, arr.size() - 1);
    }
private:
    vector<int> quickSort(vector<int>& arr, int k, int l, int r) {
        int i = l, j = r;
        while (i < j) {
            while (i < j && arr[j] >= arr[l]) j--;
            while (i < j && arr[i] <= arr[l]) i++;
            swap(arr[i], arr[j]);
        }
        swap(arr[i], arr[l]);
        if (i > k) return quickSort(arr, k, l, i - 1);
        if (i < k) return quickSort(arr, k, i + 1, r);
        vector<int> res;
        res.assign(arr.begin(), arr.begin() + k);
        return res;
    }
};
```



# [剑指 Offer 41. 数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

## 1. C++优先队列知识补充：

定义：`priority_queue<Type, Container, Functional>`

- `type:`数据类型
- `container:`容器类型，且必须为用数组实现的容器，例如`deque vector`，不可以用`list`
- `functional:`比较形式，当需要用自定义的数据类型时才需要传入这三个参数，使用基本数据类型时，只需要传入数据类型，默认是大顶堆
  - 大顶堆，即降序序列：`priority_queue <int,vector<int>,less<int> >q;`
  - 小顶堆，即升序序列：`priority_queue <int, vector<int>, greater<int> > q;`

## 2. 解法：

**维护一个大根堆和一个小根堆        **

**大根堆存放比较小的那一部分数        **

**小根堆存放比较大的那一部分数        **

**则大根堆堆顶和小根堆堆顶的元素便是中间元素        **

**当数量为奇数时，大根堆堆顶元素便是中位数（始终保持大根堆元素比小根堆多1），        **

**为偶数时，两堆顶元素之和/2.0便是中位数，(注意要除以2.0，而不是2)**

```c++
class MedianFinder {
public:
    /** initialize your data structure here. */
    priority_queue<int>  max_heap;
    priority_queue<int,vector<int>,greater<int>> min_heap;

    MedianFinder() {

    }

    /*
        将元素直接添加到大根堆中,
        若大根堆堆顶大于小根堆堆顶（即逆序），则将两堆顶元素交换
        若大根堆元素数量过多，则将堆顶元素转到小根堆中
    */
    void addNum(int num) {
        max_heap.push(num);
        if(min_heap.size() && max_heap.top() > min_heap.top()){
            auto temp = max_heap.top();
            max_heap.pop();
            max_heap.push(min_heap.top());
            min_heap.pop();
            min_heap.push(temp);
        }
        if(max_heap.size() - min_heap.size() > 1)
        {
            min_heap.push(max_heap.top());
            max_heap.pop();
        }
    }
    
    double findMedian() {
        if(max_heap.size() + min_heap.size() & 1) return max_heap.top();
        return (max_heap.top() +min_heap.top()) / 2.0;
    }
};
```



# [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

## 1. 贪心：

局部最优：当前“连续和”为负数的时候立刻放弃，从下一个元素重新计算“连续和”，因为负数加上下一个元素 “连续和”只会越来越小。

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = INT_MIN , s= 0;
        for(auto x : nums)
        {
            if(s < 0) s = 0;
            s += x;
            res = max(res,s);
        }
        return res;
    }
};
```

## 2. 动态规划

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        vector<int> dp(nums.size(), 0);
        dp[0] = nums[0];
        int res = nums[0];
        for(int i = 1; i < nums.size(); i++)
        {
            if(nums[i] + dp[i-1] < 0 ) dp[i] = max(nums[i],nums[i] + dp[i-1]);
            else
                dp[i] = max(dp[i-1] + nums[i],nums[i]);

            if (dp[i] > res) res = dp[i];
        }
        return res;
    }
};
```



# [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

 **找规律：**

**将某数按十进制位( 个 十 百 千...... )分为三部分，`high(高位)、cur(当前位)、low(低位)`**

- `cur == 0`时，1出现的次数为`high * digit`
- `cur == 1`时，1出现的次数为`high * digit + low - 1`
- `cur`为其它数字时，1出现的次数为`(high + 1) * digit`

**举个栗子：**

取`2301， cur = 0, high = 23, low = 1, 求十位的 1 出现次数(digit = 10)`

此时 1 出现在 `0010 ~ 2219`中，即出现`(22 + 1) * (9 + 1) = 230次，即 high * digit 次`



取`2314， cur = 1, high = 23, low = 4, 求十位的 1 出现次数(digit = 10)`

此时 1 出现在 `0010 ~ 2314`中，其实可以拆解为两部分

- `0010 ~ 2219`
- `2310 ~ 2314`

第一部分就是`cur = 0`时的情况，即`high * digit`次

第二部分中 1 出现的次数 为 `low + 1`

**故综上，`cur == 1`时，1出现的次数为`high * digit + low - 1`**



取`2324， cur = 2, high = 23, low = 4, 求十位的 1 出现次数(digit = 10)`

此时 1 出现在 `0010 ~ 2319`中，即出现`(23 + 1) * (9 + 1) = 240次，即(high + 1) * digit`

`cur = 3, 4 ,5,...,9`时与上同理

```c++
class Solution {
public:
    int countDigitOne(int n) {
        int res = 0;
        long digit = 1;
        int high = n / 10, cur = n % 10, low = 0;
        while(high != 0 || cur != 0)
        {
            if(cur == 0) res += high * digit;
            else if(cur == 1)
                res += high * digit + low + 1;
            else    
                res += (high + 1) * digit;

            low += cur * digit;
            digit *= 10;
            cur = high % 10;
            high /= 10;
        } 
        return res;
    }
};
```



# [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

**向上取整：**

两个int类型的整数向上取整可以用`(n + 1) / m + 1`表示

```c++
class Solution {
public:
/*
    0 ~ 9       10个 1 位
    10 ~ 99     90个 2 位
    100 ~ 999   900  3 位
*/
    int findNthDigit(int n) {   
        long long s = 9, digit = 1, i = 1;
        //确定几位数
        while(n > i * s)
        {
            n -= s * i;
            s *= 10;
            i++;
            digit *= 10;
        }
        //确定是哪个数
        int number = digit + (n-1)/i + 1 - 1;  //向上取整

        //确定具体是哪一位
        int r = n % i ? n % i : i;
        for(int j = 0; j < i - r; j++)
        {
            number /= 10;
        }
        return number % 10;

    }
};
```



# [剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

**自定义排序**

对于数组中的数字a , b, 其对应字符为A，B

若满足`A + B < B + A`,则认为`A < B`

依据上述规则对数组排序

```c++
class Solution {
public:
    string minNumber(vector<int>& nums) {
        sort(nums.begin(),nums.end(),[&](int a, int b){
            string A = to_string(a);
            string B = to_string(b);
            return A + B < B + A;
        });

        string res = "";
        for(auto num : nums){
            res += to_string(num);
        }
        return res;
    }
};
```



# [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

**动态规划，踩了不少坑啊，不够细心**

首先将数字转换为字符串string

- `dp[i]：前 i 位数字的翻译方法数量`

- 状态转移方程：

  - 可以分为两种状态，第一种是单独翻译第 i 个数字，第二种方法是将 第 i 位数字与第 i-1位数字一起翻译

  - 单独翻译的话，状态转移方程就是`dp[i] = dp[i-1]`

  - 一起翻译的话，首先得判断第 i 位数字与第 i-1位数字组成的两位数是否在可翻译范围内

    - ```c++
      string temp = s.substr(i-2,2);
      int number = stoi(temp);
      if(number >= 0 && number <= 25)
      ```

    - 同时，还必须要判断第 i-1位数字是否为'0'，**这也是我踩的一个大坑**，              `if(s[i-2] != '0' && number >= 0 && number <= 25)`

    - 状态转移方程：`dp[i] += dp[i-2];`

```c++
class Solution {
public:
    int translateNum(int num) {
        string s = to_string(num);
        vector<int> dp(s.size()+1,0);
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= s.size(); i++)
        {
            dp[i] = dp[i-1];
            string temp = s.substr(i-2,2);
            int number = stoi(temp);
            if(s[i-2] != '0' && number >= 0 && number <= 25)
            {
                dp[i] += dp[i-2];
            }
        }
        return dp[s.size()];
    }
};
```



# [剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

## 1. 动态规划：

```c++
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        if(grid.empty() || grid[0].empty()) return 0;
        vector<vector<int>> dp(grid.size(), vector<int>(grid[0].size(), 0));
        dp[0][0] = grid[0][0];
        for(int i = 1; i < grid.size(); i++) dp[i][0] = grid[i][0] + dp[i-1][0];
        for(int i = 1; i < grid[0].size(); i++) dp[0][i] = grid[0][i] + dp[0][i-1];

        for(int i = 1; i < grid.size(); i++){
            for(int j = 1; j < grid[0].size(); j++)
            {
                dp[i][j] = grid[i][j] + max(dp[i-1][j],dp[i][j-1]);;
            }
        }
        return dp[grid.size()-1][grid[0].size()-1];
    }
};
```

## 2. 贪心(大错特错)：

**按动态规划写出来以后，感觉有点小题大做(我也不知道为啥有这种感觉，可能因为这题太简单了？)，于是脑子一抽，觉得可以用贪心，反正每次只能往右或者往下，那我就每次取两个方向上最大的那个元素，然后如果有一边到了边界，则只能往一个方向走，大错特错！**

**动态规划中的每一个点都是最优解，而贪心只能求出一个解**



# [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

**动态规划：**

- `dp[j]: 以字符s[j]为结尾的“最长不重复子字符串”的长度`

- 转移方程：固定右边界`j`，设字符`s[j]`左边距离最近的相同字符为`s[i]`,即`s[i] == s[j]`

  - `i < 0`时，即 `s[j]`左边无相同字符，则`dp[j] = dp[j-1] + 1`

  - `dp[j-1] < j-i`，说明字符`s[i]`在区间`dp[j-1]`之外，例如

    `a b a c , dp[2] = 2,即字符串ba, 而s[i] = s[0] = a,这个a字符在ba之前`

    则`dp[j] = dp[j-1] + 1`

  - `dp[j-1] ≥ j-i`，说明字符`s[i]`在区间`dp[j-1]`之中，例如

​					`a b a c, dp[1] = 2,即字符串ab，而当j == 2时，s[i] == s[0] == a,`

​					`此时dp[j-1] = dp[1] = 2 >= 2 - 0`,则`dp[j]`的左边界由`s[i]`决定，

​					即`dp[j] = j - i`

**返回max(dp)**

## 1. 动态规划 + 哈希表

遍历字符串时，**利用哈希表存储各字符最后一次出现的索引位置**

遍历到`s[j]`时，可通过访问哈希表来获取最近的相同字符的索引

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.empty()) return 0;
        unordered_map<char,int> hash;
        int res = 0, temp = 0;
        for(int j = 0; j < s.size(); j++)
        {   
            if(hash.find(s[j]) != hash.end())
            {
                int i = hash[s[j]];
                temp = temp < j - i ? temp + 1 : j - i;
            }else
                temp++;
                
            res = max(res,temp); 
            hash[s[j]] = j;
        }
        return res;
    }
};
```

## 2. 滑动窗口

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.size() == 0) return 0;
        unordered_set<char> hash;
        int maxStr = 0;
        int left = 0;
        for(int i = 0; i < s.size(); i++){
            while (hash.find(s[i]) != hash.end()){
                hash.erase(s[left]);
                left++;
            }
            maxStr = max(maxStr,i-left+1);
            hash.insert(s[i]);
    	}
        return maxStr;
    }
};
```



# [剑指 Offer 49. 丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

```c++
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> dp(1,1);
        int i = 0, j = 0, k = 0;
        while(--n)
        {
            int t = min(dp[i]*2, min(dp[j]*3, dp[k]*5));
            dp.push_back(t);
            if(t == dp[i]*2) i++;
            if(t == dp[j]*3) j++;
            if(t == dp[k]*5) k++;
        }
        return dp.back();
    }
};
```



# [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

**哈希表**

```c++
class Solution {
public:
    char firstUniqChar(string s) {
        if(s.empty()) return ' ';
        unordered_map<char,int> hash;
        for(auto x : s) hash[x]++;
        for(auto x : s){
            if(hash.find(x) != hash.end())
                if(hash[x] == 1)
                    return x;
        }
        return ' ';
    }
};
```



```c++
class Solution {
public:
    char firstUniqChar(string s) {
        unordered_map<int,bool> hash;
        for(auto x : s){
            hash[x] = hash.find(x) == hash.end();
        }
        for(auto x : s){
            if(hash[x] == true) return x;
        }
        return ' ';
    }
};
```



# [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

## 归并排序回顾：

![1024555-20161218163120151-452283750](F:\A3-git_repos\数据结构与算法_notes\图片\1024555-20161218163120151-452283750.png)

## 解法：

在归并的合并过程中，对于两个数组A B，可知这两个数组均为升序数组，因此，若此时`A[i]>B[j]`

则`{A[i],B[j]}`为一个逆序对，因为为升序数组，因此`数组A中从 i 至末尾所有元素，均可与B[j]构成一个逆序对`，因此，我们可以在归并过程中完成逆序对的统计工作

```c++
class Solution {
public:
    int reversePairs(vector<int>& nums) {
        vector<int> temp(nums.size() + 1,0);
        return mergeSort(0,nums.size()-1, nums,temp);
    }

    int mergeSort(int l, int r, vector<int>& nums, vector<int>& temp)
    {
        if(l >= r) return 0;
        int m = (l + r) / 2;
        int res = mergeSort(l, m, nums, temp) + mergeSort(m + 1, r ,nums, temp);

        for(int k = l; k <= r; k++)
            temp[k] = nums[k];

        int i = l, j = m + 1, k = l;
        while(i <= m && j <= r){
            if(temp[i] <= temp[j]) nums[k++] = temp[i++];
            else{
                nums[k++] = temp[j++];
                res += m - i + 1;
            }
        }
        while(i <= m) nums[k++] = temp[i++];
        while(j <= r) nums[k++] = temp[j++];

        return res;   
    }
};
```



# [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA || !headB) return nullptr;
        ListNode* curA = headA;
        ListNode* curB = headB;

        while(curA != curB)
        {
            if(curA) curA = curA->next;
            else curA = headB;
            if(curB) curB = curB->next;
            else curB = headA;
        }
        return curB;
    }
};
```

## 注意事项：

**在循环体中，我一开始写的是`if(curA->next) curA = curA->next;`，咋一看好像没有问题，可是当输入的两个链表不相交时就报错了，程序会陷入死循环，因此采用`if(curA) curA = curA->next`写法，循环一定会终止，因为就算不相交，经过一次交换以后，最终会`curA = curB = nullptr`**

# [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

**二分法**

对于给定区间`[i,j]`

- `int mid = i + j >> 1;`
- `if(nums[mid] < target)`
  - 则说明`target在区间[mid+1, j]中`，因此执行`l = mid + 1`
- `if(nums[mid] < target)`
  - 则说明`target在区间[i, mid-1]中`，因此执行`r = mid - 1`
- `if(nums[mid] == target)`
  - 则右边界在区间`[mid+1，j]`中，因此执行`l = mid + 1`
  - 则左边界在区间`[i, mid-1]`中，因此执行`r = mid - 1`



```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(nums.empty()) return 0;
        int left = 0, right = 0;
        int l = 0, r = nums.size() - 1;
        //右边界
        while(l <= r){
            int mid = (l + r) / 2;
            if(nums[mid] > target) r = mid - 1;
            else l = mid + 1;
        }
        right = l;
        //若数组中无target
        if(r >= 0 && nums[r] != target) return 0;
        //左边界
        l = 0, r = nums.size() - 1;
        while(l <= r){
            int mid = (l + r) / 2;
            if(nums[mid] >= target) r = mid - 1;
            else l = mid + 1;
        }
        left = r;

        return right - left - 1;
    }
};
```



# [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

**二分**

**当`nums[mid] == mid`即数组值等于下标值时，说明从起始位置到mid的数组部分不缺元素，缺失的元素一定在[mid+1,j]中**

```c++
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        if(nums.empty()) return -1;
        int l = 0, r = nums.size() -1 ;
        while(l < r)
        {
            int mid = l + r >> 1;
            if(nums[mid] == mid) l = mid + 1;
            else{
                r = mid - 1;
            }
        }
        if(nums[l] == l) return l + 1;
        return l;
    }
};
```



# [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

**中序倒序遍历**

```c++
class Solution {
public:
    int res;
    void dfs(TreeNode* root, int& k)
    {
        if(!root){
            return;
        }
        dfs(root->right,k);

        if(k == 0) return;
        if(--k == 0) res = root->val;

        dfs(root->left,k);

    }
    int kthLargest(TreeNode* root, int k) {
        if(!root) return -1;
        dfs(root,k);
        return res;
    }
};
```



# [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
```



# [剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }

    bool isBalanced(TreeNode* root) {
        if(!root) return true;
        int left = maxDepth(root->left);
        int right = maxDepth(root->right);
        return (abs(left - right) <= 1) && isBalanced(root->left) && isBalanced(root->right);
    }
};
```



# [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

```c++
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        //相同的数异或运算结果为0，将整个数组进行异或运算，则最终结果为两个不重复数字的异或
        int aXORb = 0;
        for(auto x : nums)  aXORb ^= x;

        //aXORb的二进制表示中，为1的那一位代表两数中有一个数该位为1，另一个书该位为0
        //从低位开始，找出aXORb中第一个为1的是第k位，由此可将数组分为两部分
        //一部分是第k位为1的数，另一部分是第k位为0的数
        //再对两部分元素之一进行异或，可得到一个结果
        int k = 0;
        while(!(aXORb >> k & 1)) k++;

        int first = 0;
        for(auto x : nums){
            if(x >> k & 1) first ^= x;
        }

        int second = aXORb ^ first;
        return vector<int>{first,second};
    }
};
```



# [剑指 Offer 56 - II. 数组中数字出现的次数 II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

**除了一个数A以外，其他数都出现三次，则对于32位的整数，统计数组中所有数的每位出现的次数，则出现次数为3的位数，在A中该位为0**

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        vector<int> number(32,0);
        for(auto x: nums)
        {
            for(int i = 0; i < 32; i++){
                if((x >> i) & 1) number[i]++;
            }
        }
        int res = 0;
        for(int i = 0; i < 32; i++){
            if(number[i] % 3 == 1)
                res |= (1 << i);
        }
        return res;
    }
};
```



# [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

**双指针**

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        if(nums.empty()) return res;
        int i = 0, j = nums.size() - 1;
        while(i < j)
        {
            if(nums[i] + nums[j] < target) i++;
            else if(nums[i] + nums[j] > target) j--;
            else
                return vector<int>{nums[i],nums[j]};
        }
        return vector<int>{ };
    }
};
```



# [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

**滑动窗口 双指针**

```c++
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> res;
        int i = 1, j = 1, sum = i;
        while(i <= target / 2)
        {
            if(sum < target)
            {
                j++;
                sum += j;
            }else if(sum > target)
            {
                sum -= i;
                i++;
            }else{
                vector<int> temp;
                for(int k = i; k <= j; k++){
                    temp.push_back(k);
                }
                res.push_back(temp);
                sum -= i;
                i++;
            }
        }
        return res;
    }
};
```



# [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

**字符串模拟，从后往前分割单词**

```c++
class Solution {
public:
    string reverseWords(string s) {
        if(s.empty()) return s;
        int right = s.size() - 1;
        string res;
        while(right >= 0)
        {
            //从后往前找到第一个字母
            while(right >= 0 && s[right] == ' ') right--;
            if(right < 0) break;
            //找到第一个单词
            int left = right;
            while(left >= 0 && s[left] != ' ') left--;  //此时left指向一个单词的前面的空格
            //分割出单词
            res += s.substr(left+1,right - left);
            //添加空格
            res += ' ';

            //继续向前分割
            right = left;
        }
        //去除尾部空格
        if(!res.empty()) res.pop_back();
        return res;
    }
};
```



# [剑指 Offer 58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

```c++
class Solution {
public:
    string reverseLeftWords(string s, int n) {
        reverse(s.begin(),s.end());
        reverse(s.begin(),s.end() - n);
        reverse(s.begin() + s.size() - n, s.end());
        return s;
    }
};
```



# [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

**双端队列**

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        if(nums.empty()) return vector<int>{ };
        vector<int> res;
        deque<int> dq;
        int lhs = 0, rhs = 0;
        while(rhs < nums.size())
        {
            while(!dq.empty() && nums[rhs] > nums[dq.back()]) dq.pop_back();
            dq.push_back(rhs);
            
            if(rhs > k - 1){
                lhs++;
                if(lhs > dq.front()) dq.pop_front();
            }
            if(rhs >= k - 1){
                res.push_back(nums[dq.front()]);
            }
            rhs++;
        }
        return res;
    }
};
```



# [剑指 Offer 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

**这不敢敢单单**

```c++
class MaxQueue {
public:
    MaxQueue() {

    }
    
    int max_value() {
        if(q.empty()) return -1;
        return dq.front();
    }
    
    void push_back(int value) {
        if(q.empty())
        {
            dq.push_back(value);
        }
        else{
            while(!dq.empty() && value > dq.back()) dq.pop_back();
            dq.push_back(value);
        }
        q.push(value);
    }
    
    int pop_front() {
        if(q.empty()) return -1;
        int cur = q.front();
        q.pop();
        if(cur == dq.front()) dq.pop_front();
        return cur;
    }
private:
    deque<int> dq;
    queue<int> q;
};
```



# [剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

## **动态规划：**

- `dp[i][j]:投掷完 i 枚骰子后，点数 j 的出现次数（j为所有骰子的点数和）`

- 状态转移:

  - ```c++
    for (第n枚骰子的点数 i = 1; i <= 6; i ++) {
        dp[n][j] += dp[n-1][j - i]
    }
    ```

- 边界处理：

  - ```c++
    for (int i = 1; i <= 6; i ++) {
        dp[1][i] = 1;
    }
    ```

    


```c++
class Solution {
public:
    vector<double> dicesProbability(int n) {
        int base = pow(6,n);
        vector<vector<double>> dp(n + 1,vector<double>(6 * n + 1,0));
        for(int i = 1; i <= 6; i++) dp[1][i] = 1;
        for(int i = 2; i <= n; i++)
        {
            for(int j = i; j <= 6 * n; j++)
            {
                for(int k = 1; k <= 6; k++)
                {
                    if(k >= j) break;
                    dp[i][j] += dp[i-1][j-k];
                }
            }
        }

        vector<double> res;
        for(int i = n; i <= 6 * n; i++){
            res.push_back(dp[n][i] / base);
        }
        return res;
    }
};
```

## 空间优化：

**利用滚动数组，此时遍历顺序必须从后往前，这样才能利用上一层的结果**

- `dp[i]:点数 i 的出现次数`

- 状态转移：

  - ```c++
    for(int j = 1; j <= 6; j++)
    	dp[i] += dp[i-j];
    ```

- 初始化：

  - ```c++
    for (int i = 1; i <= 6; i ++) {
    	dp[i] = 1;
    }
    ```

```c++
class Solution {
public:
    vector<double> dicesProbability(int n) {
        int base = pow(6,n);
        vector<double> dp(6 * n + 1);
        for(int i = 1; i <= 6; i++) dp[i] = 1;
       
        for (int i = 2; i <= n; i ++) {
            for (int j = 6*i; j >= i; j --) {
                dp[j] = 0;
                for (int cur = 1; cur <= 6; cur ++) {
                    if (j - cur < i-1) {
                        break;
                    }
                    dp[j] += dp[j-cur];
                }
            }
        }
        vector<double> res;
        for(int i = n; i <= 6 * n; i++){
            res.push_back(dp[i] / base);
        }
        return res;
    }
};
```



# [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

- 无重复元素
- 除0以外，最大值 - 最小值 < 5

```c++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        set<int> repeat;
        int min = 14, max = 0;
        for(auto x : nums){
            if(x == 0) continue; // 跳过大小王
            min = min > x ? x : min;
            max = max < x ? x : max;
            if(repeat.find(x) != repeat.end()) return false;
            repeat.insert(x);
        }

        return max - min < 5;
    }
};
```



# [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)



# [剑指 Offer 63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

## 1. 贪心 ： 在最低点买入，最高点卖出

**遍历数组，维护一个最低价格，在每一天都假设卖出，比较利润，最后返回最大利润**

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.empty()) return 0;
        int MIN = INT_MAX;
        int res = INT_MIN;
        for(int i = 0; i < prices.size(); i++)
        {
            if(MIN > prices[i]) MIN = prices[i];
            res = max(prices[i]-MIN,res);
        }
        return res;
    }
};
```

## 2. 动态规划：

- `dp[i]:前 i 天股票的最大利润`
- 状态转移：
  - `dp[i] = max(dp[i - 1], nums[i-1] - min(prices[i]))`

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.empty()) return 0;
        vector<int> dp(prices.size() + 1, 0);

        int Min = INT_MAX;
        for(int i = 1; i <= prices.size(); i++)
        {
            Min = min(Min,prices[i-1]);
            dp[i] = max(dp[i-1],prices[i-1] - Min);
        }
        return dp[prices.size()];

    }
};
```



# [剑指 Offer 64. 求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

**传统递归**

```c++
class Solution {
public:
    int sumNums(int n) {
        int res = n;
        if(n == 1) return res;
        res += sumNums(n - 1);
        return res;
    }
};
```

**利用`&&`代替 `if`用来终止递归**

```c++
class Solution {
public:
    int sumNums(int n) {
        int res = n;
        (n > 1) && (res += sumNums(n-1));
        return res;
    }
};
```



# [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

`^ 异或` ----相当于 无进位的求和， 想象10进制下的模拟情况：（如:19+1=20；无进位求和就是10，而非20；因为它不管进位情况）

`& 与` ----相当于求每位的进位数， 先看定义：1&1=1；1&0=0；0&0=0；即都为1的时候才为1，正好可以模拟进位数的情况,还是想象10进制下模拟情况：（9+1=10，如果是用&的思路来处理，则9+1得到的进位数为1，而不是10，所以要用<<1向左再移动一位，这样就变为10了）；

**一直循环计算无进位和以及进位值，直至进位值为0，此时直接输出无进位和**

```c++
class Solution {
public:
    int add(int a, int b) {
        //因为不允许用+号，所以求出异或部分和进位部分依然不能用+ 号，所以只能循环到没有进位为止        
        while(b!=0)
        {
        //保存进位值，下次循环用
            int c=(unsigned int)(a&b)<<1;//C++中负数不支持左移位，因为结果是不定的
        //保存不进位值，下次循环用，
            a^=b;
        //如果还有进位，再循环，如果没有，则直接输出没有进位部分即可。
            b=c;   
        }
        return a;
    }
};
```



# [剑指 Offer 66. 构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

**两次遍历**

- 从前往后遍历数组，计算对应元素的前面所有元素之积
- 从后往前遍历数组，将在对应元素后面的元素之积乘入数组

```c++
class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        if(a.empty()) return vector<int>{};
        vector<int> res(a.size(),1);
        for(int i = 1; i < a.size(); i++)
        {
            res[i] = res[i-1] * a[i-1];
        }
        for(int i = a.size() - 2; i >= 0; i--)
        {
            res[i] *= a[i+1];
            a[i] *= a[i+1];
        }
        return res;
    }
};
```



# [剑指 Offer 67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

**字符串模拟**

```c++
class Solution {
public:
    int strToInt(string s) {
        if(s.empty()) return 0;
        long res = 0;
        int i = 0;
        while(s[i] == ' ') i++;
        if(i >= s.size()) return 0;
        s = s.substr(i,s.size() - i + 1);

        bool flag = true;
        if(s[0] == '-') flag = false;

        for(int i = 0; i < s.size(); i++)
        {
            if(s[0] == '+' || s[0] == '-'){
                if(s.size() == 1) return 0;
                int j = i + 1;
                while(s[j] >= '0' && s[j] <= '9' && j < s.size())
                {
                    if(res > INT_MAX / 10) return flag ? INT_MAX : INT_MIN;
                    res = res * 10 + s[j] - '0';
                    j++;
                }
                if(res > INT_MAX) return flag ? INT_MAX : INT_MIN;
                return flag ? res : -res;
            }  
            else if(s[0] < '0' || s[0] > '9') return 0;
            else{
                while(s[i] >= '0' && s[i] <= '9' && i < s.size())
                {
                    if(res > INT_MAX) return flag ? INT_MAX : INT_MIN;
                    res = res * 10 + s[i] - '0';
                    i++;
                }
                if(res > INT_MAX) return flag ? INT_MAX : INT_MIN;
                return flag ? res : -res;
            }
        }
        return res;
    }
};
```

**精简一下：**

```c++
class Solution {
public:
    int strToInt(string str) {
        int i = 0;
        while(i < str.size() && str[i] == ' ') i++;
        long long res= 0;
        int _minus = false;
        if(str[i] == '+') i++;
        else if(str[i] == '-'){
            i++;
            _minus = true;
        }
        while(i < str.size() && str[i] >= '0' && str[i] <= '9')
        {
            if(res > INT_MAX){
                if(_minus) return INT_MIN;
                return INT_MAX;
            }
            res = res * 10 + str[i]-'0';
            i++;
        }
        if(res > INT_MAX){
            if(_minus) return INT_MIN;
            return INT_MAX;
        }
        if(_minus) return res *= -1;
        return res;
    }
};
```



# [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

**二叉搜索树，左子节点值 < 根节点值 < 右子节点值**

- 如果两节点分列在当前节点两端，则当前节点就是最近公共祖先
- 如果当前节点就是两节点之一，则当前节点就是最近公共祖先，即两节点位于同一根树枝上
- 否则，递归的去左右子节点寻找最近公共祖先

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return root;
        if(root == p || root == q) return root == p ? p : q;
        if(p->val > root->val && q->val < root->val || p->val < root->val && q->val > root->val) return root;
        TreeNode* res = lowestCommonAncestor(root->left,p,q);
        if(!res) return lowestCommonAncestor(root->right,p,q);
        return res;
    }
};
```





# [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

**与二叉搜索树不同，不可以直接通过比较节点值的大小来判断当前节点的位置，因此，直接记录下分别在左右子树搜寻的结果：**

- 如果两结果之一为空，说明两节点位于根节点的同一侧，则不为空的那个结果就是最近公共祖先
- 如果两结果都不为空，说明两节点位于根节点两侧，则根节点就是最近公共祖先

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root) return root;
        if(root == p || root == q) return root == p ? p : q;
        TreeNode* left = lowestCommonAncestor(root->left, p , q);
        TreeNode* right = lowestCommonAncestor(root->right, p , q);
        if(!left) return right;
        if(!right) return left;
        else return root;
    }
};
```

