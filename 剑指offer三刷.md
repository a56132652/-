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



























