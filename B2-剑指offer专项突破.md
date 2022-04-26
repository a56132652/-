# 位运算

## [剑指 Offer II 001. 整数除法](https://leetcode-cn.com/problems/xoh6Oh/)

举个例子：11 除以 3 。
首先11比3大，结果至少是1， 然后我让3翻倍，就是6，发现11比3翻倍后还要大，那么结果就至少是2了，那我让这个6再翻倍，得12，11不比12大，吓死我了，差点让就让刚才的最小解2也翻倍得到4了。但是我知道最终结果肯定在2和4之间。也就是说2再加上某个数，这个数是多少呢？我让11减去刚才最后一次的结果6，剩下5，我们计算5是3的几倍，也就是除法，看，递归出现了

```c++
class Solution {
public:
    int divide(int dividend, int divisor) {
        if(dividend == 0) return 0;
        if(divisor == 1) return dividend;
        if(divisor == -1){
            if(dividend>INT_MIN) return -dividend;// 只要不是最小的那个整数，都是直接返回相反数就好啦
            return INT_MAX;// 是最小的那个，那就返回最大的整数啦
        }
        long a = dividend;
        long b = divisor;
        int sign = 1; 
        if((a>0&&b<0) || (a<0&&b>0)){
            sign = -1;
        }
        a = a>0?a:-a;
        b = b>0?b:-b;
        long res = div(a,b);
        if(sign>0)return res>INT_MAX?INT_MAX:res;
        return -res;
    }
    int div(long a, long b){  // 似乎精髓和难点就在于下面这几句
        if(a<b) return 0;
        long count = 1;
        long tb = b; // 在后面的代码中不更新b
        while((tb*2)<=a){
            count *= 2; // 最小解翻倍
            tb*= 2; // 当前测试的值也翻倍
        }
        return count + div(a-tb,b);
    }
};
```



## [剑指 Offer II 002. 二进制加法](https://leetcode-cn.com/problems/JFETK5/)

**简单一个小模拟**

```c++
class Solution {
public:
    string addBinary(string a, string b) {
        int na = a.size();
        int nb = b.size();
        string res = "";
        int i = na - 1, j = nb - 1;
        int cnt = 0;
        while(i >= 0 || j >= 0){
            int sa = i < 0 ? 0 : a[i] - '0';
            int sb = j < 0 ? 0 : b[j] - '0';
            i--;
            j--;
            int sum = sa + sb + cnt;
            int ans = sum % 2;
            cnt = sum / 2;

            res += (ans + '0');

        }
        if(cnt != 0)
            res += (cnt + '0');

        reverse(res.begin(), res.end());
        return res;
    }
};
```



## [剑指 Offer II 003. 前 n 个数字二进制中 1 的个数](https://leetcode-cn.com/problems/w3tCBm/)

```c++
class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> res ;
        for(int i = 0; i <= n; i++){
            int cnt = 0;
            int cur = i;
            while(cur){
                if(cur & 1) cnt++;
                cur >>= 1;
            }
            res.push_back(cnt);
        }
        return res;
    }
};
```

**动态规划**

- 对于一个正整数 `x` , 存在一个正整数`y`,且`y`是`2`的整数次幂，即 `y `的二进制表示中只有最高位为 `1`
- 因此，`x `的二进制表示中` 1 `的个数为 `x-y `的二进制表示中 `1 `的个数 + `1`
- 即`dp[i] = dp[i-j] + 1`
- 对于某数是否为`2`的整数次幂，可以利用如下方法判断
  - 若`x  & (x-1) == 0`，则说明 `x`是`2`的整数次幂

**代码：**

```c++
class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> dp(n+1,0);
        int flag = 0;
        for(int i = 1; i <= n; i++){
            //实时记录最近的2的整数次幂
            //要特别注意运算符的优先级，这里 $ 运算的优先级低于 ==，因此要用括号阔起来
            if((i & (i-1)) == 0)
                flag = i;
            dp[i] = dp[i-flag] + 1;
        }
        return dp;
    }
};
```



## [剑指 Offer II 004. 只出现一次的数字 ](https://leetcode-cn.com/problems/WGki4K/)

**比特位计数，将所有数字的每个比特位的数字存储在一个数组中，若某一比特位数字为3的倍数，则说明只出现了一次的数字的该比特位为0，否则为1，由此可以计算出该数字**

代码：

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        vector<int> cnt(32,0);
        for(auto num : nums){
            for(int i = 0; i < 32; i++){
                cnt[i] += ((num >> i) & 1);
            }
        }

        for(int i = 0; i < 32; i++){
            if(cnt[i] % 3) res |= (1 << i);
        }
        return res;
    }
};
```



## [剑指 Offer II 005. 单词长度的最大乘积](https://leetcode-cn.com/problems/aseY1I/)

该题暴力法就是每两个单词进行比较，我们可以利用**位运算**对单词比较的过程进行优化

- 因为题目给定的单词均由小写字母组成，因此我们将每一个单词中的字母进行映射操作，**映射到一个INT类型整数上**

  - 单词从**左到右**为 低位 到 高位
  - 数组从**右到左**为 低位 到 高位

- 对于一个单词，记录每一位是否出现，例如给定`abc`与`add`

  - 对于`abc`则将其映射为二进制表示则为`000...00111`，该二进制一共26位，前23位为0，低位3个为1，表示`a` `b` `c`三个字符都出现在了该单词中

  - 同理，对于`add`,将其映射为二进制表示则为`000...01001`

  - 两者相与结果不为 0 ，说明两单词有重复字母

  - 映射字母代码为

    ```c++
        int convert(string s){
            int res = 0;
            for(int i = 0; i < s.size(); i++){
                res |= 1 << (s[i] - 'a');
            }
            return res;
        }
    ```

**完整代码如下：**

```c++
class Solution {
public:
    int convert(string s){
        int res = 0;
        for(int i = 0; i < s.size(); i++){
            res |= 1 << (s[i] - 'a');
        }
        return res;
    }
    int maxProduct(vector<string>& words) {
        int n = words.size();
        vector<int> masks(n);
        //对每个单词计算映射值
        for(int i = 0; i < n ; i++){
            masks[i] = convert(words[i]);
        }

        int res = 0;
        for(int i = 0; i < n; i++){
            for(int j = i + 1; j < n; j++){
                //注意细节，& 的优先级低于 == ，因此与操作表达式要额外加括号
                if((masks[i] & masks[j]) == 0){
                    //注意细节，max()函数要求两参数的类型相同，因此对乘积结果进行强转
                    //.size()函数返回值为 unsigned int
                    res = max(res, int(words[i].size() * words[j].size()));
                }
            }
        }
        return res;
    }
};
```



# 双指针（滑动窗口）

## [剑指 Offer II 006. 排序数组中两个数字之和](https://leetcode-cn.com/problems/kLl5u1/)

**简单双指针**

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        vector<int> res;
        int i = 0;
        int j = numbers.size() - 1;
        while(i < j){
            if(numbers[i] + numbers[j] > target)
                j--;
            else if(numbers[i] + numbers[j] < target)
                i++;
            else{
                res.push_back(i);
                res.push_back(j);
                return res;
            }
        }
    return res;
    }
};
```

## [剑指 Offer II 007. 数组中和为 0 的三个数](https://leetcode-cn.com/problems/1fGaJU/)

**该题不难，但是有几个要注意的点**

1. 题目要求不能出现重复结果，要从以下几点考虑去重
   1. 首先是最外层的循环，如果发现当前数字与上一层循环数字相同，则跳过
   2. 其次是内层循环中，找到一组结果后，将左右指针移动之后，若发现新值与旧值相同，则跳过，继续移动做右指针

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        for(int i = 0; i < nums.size(); i++){
            if(i >= 1 && nums[i] == nums[i-1]) continue;
            int l = i + 1, r = nums.size() - 1;
            while(l < r){
                if(nums[i] + nums[l] + nums[r] == 0){
                    res.push_back(vector<int>{nums[i], nums[l++], nums[r--]});
                    while(l < r && nums[l] == nums[l-1]) l++;
                    while(l < r && nums[r] == nums[r+1]) r--;
                }else if(nums[i] + nums[l] + nums[r] < 0){
                    l++;
                    while(l < r && nums[l] == nums[l-1]) l++;
                }else{
                    r--;
                    while(l < r && nums[r] == nums[r+1]) r--;
                }
            }
        }
        return res;
    }
};
```



## [剑指 Offer II 008. 和大于等于 target 的最短子数组](https://leetcode-cn.com/problems/2VG8Kg/)

### **双指针滑动窗口**

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int left = 0;
        int sum = 0;
        int res = INT_MAX;
        for(int i = 0; i < nums.size(); i++){
            sum += nums[i];
            while(sum >= target){
                res = res > (i - left + 1) ? (i - left + 1) : res;
                sum -= nums[left++];
            }          
        }
        return res == INT_MAX ? 0 : res;
    }
};
```

**贴上我的错误代码**

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int i = 0, j = 0;
        int res = INT_MAX;
        int sum = 0;
        while(j < nums.size()){
            while( j < nums.size() && sum < target){
                sum += nums[j];
            }
            while(i < nums.size() && sum >= target){
                res = min(j - i + 1, res);
                sum -= nums[i++];
            }
        }
        return res == INT_MAX ? 0 : res;
    }
};
```

- 提交之后发现，所有结果都比答案多 1 

- 在内层循环中，我对右指针的操作其实是多余的，这样操作会导致 右指针最终指向数组最后一个元素的下一个元素，即 `j = nums.size()`,因此导致最终结果始终比正确结果大 1

  ```c++
              while( j < nums.size() && sum < target){
                  sum += nums[j];
              }
  ```

**正确代码**

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int i = 0, j = 0;
        int res = INT_MAX;
        int sum = 0;
        while(j < nums.size()){
            
            sum += nums[j];
            while(i < nums.size() && sum >= target){
                res = min(j - i + 1, res);
                sum -= nums[i++];
            }
            j++;
        }
        return res == INT_MAX ? 0 : res;
    }
};
```

### 前缀和加二分查找

**要想实现`nlogn`的时间复杂度，可以利用前缀和加二分查找**

- 前缀和数组`sum`表示数组`nums`中前 `i`个元素的和

  ```c++
  sum[i] = sum[i-1] + nums[i-1]
  ```

- 因为`nums`中元素均大于0，因此前缀数组是一个递增的数组

  - 对于`sum[i]`，利用二分法查找最近的`sum[l]`,使得`sum[l] - sum[i] >= target`

**代码：**

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = nums.size();
        vector<int> sum(n + 1, 0);
        int res = INT_MAX;
        for(int i = 1; i < sum.size(); i++){
            sum[i] = sum[i-1] + nums[i-1];
        }
        for(int i = 1; i <= n; i++){
            int t = sum[i-1] + target;
            int l = 1, r = n;
            while(l <= r){
                int mid = (l + r) >> 1;
                if(sum[mid] >= t) r = mid - 1;
                else l = mid + 1;
            }
            if(l <= n && sum[l] >= t)
                res = min(res, l - i + 1);
        }
        return res == INT_MAX ? 0 : res;
    }
};
```



## [剑指 Offer II 009. 乘积小于 K 的子数组](https://leetcode-cn.com/problems/ZVAVXX/)

**滑动窗口**

使用滑动窗口时要注意的点就是对当前窗口内的元素如何计算满足要求的子数组个数

**对于当前窗口，满足窗口内元素`(nums[i], nums[i+1],nums[i+2])`之积小于等于`k`**

- 则满足条件的子数组一共有
  - `nums[i]`
  - `nums[i], nums[i+1]`
  - `nums[i], nums[i+1], nums[i+2]`
- 即`right - left + 1` 个

```c++
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        if (k <= 1) return 0;
        int prod = 1, ans = 0, left = 0;
        for (int right = 0; right < nums.size(); right++) {
            prod *= nums[right];
            while (prod >= k) prod /= nums[left++];
            ans += right - left + 1;
        }
        return ans;
    }
};
```

**利用while()循环**

```c++
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        if (k <= 1) return 0;
        int n = nums.size();
        int i = 0, j = 0;
        int cur = 1;
        int res = 0;
        while(j < n){
            cur *= nums[j];
            while(i < j && cur >= k){
                cur /= nums[i++];
            }
            
            res += (j - i + 1);
            j++;
        }
        return res;
    }
};
```

## [剑指 Offer II 014. 字符串中的变位词](https://leetcode-cn.com/problems/MPnaiL/)

### 滑动窗口

碰到该题，第一想法就是滑动窗口，但是怎么利用滑动窗口是很有讲究的

- 一开始我的想法是窗口内存储字符，只要这些字符都属于另一字符串且长度相同， 那就符合条件
  - 但是实现的时候卡在了判断字符是否属于零一字符串上
- **题解中，窗口用于维护字符数量，因为根据题意，当满足条件时，窗口内元素个数与零一字符串元素个数是一致的**，妙哉
  - 窗口长度固定为`s1.size()`
  - 每次从右边进来一个字符，并且从左边出去一个字符

```c++
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        if(s1.size() > s2.size()) return false;
        int n = s1.size();
        vector<int> v1(26,0);
        vector<int> v2(26,0);
        for(int i = 0; i < s1.size(); i++){
            v1[s1[i]-'a']++;
            v2[s2[i]-'a']++;
        }
        if(v1 == v2) return true;
        int l = 0;
        for(int r = n; r < s2.size(); r++){
            v2[s2[r]-'a']++;
            v2[s2[l++]-'a']--;
            if(v1 == v2) return true;
        }
        return false;
    }
};
```

### 双指针

- 利用一个频次数组`cnt`
  - 首先利用`s1`初始化频次数组，`s1`中每出现一个字符，就将其频次**减一**
    - 初始化完成后，`cnt`内的元素之和是等于`-s1.size()`
  - 然后使用双指针遍历`s2`
    - 右指针每移动一次，就将其指向的字符的频次**加一**，`cnt`元素之和也会**加一**
      - 若其指向的字符的频次大于一，则移动左指针，并相应的将左指针指向的字符频次减一
  - 右指针每移动一次，就将其指向的字符的频次**加一**，`cnt`元素之和也会**加一**
    - 因此，当`right - left + 1 == s1.size()`时，此时`cnt`元素之和为0，因此得到一个目标子串

```c++
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        int n = s1.length(), m = s2.length();
        if (n > m) {
            return false;
        }
        vector<int> cnt(26);
        for (int i = 0; i < n; ++i) {
            --cnt[s1[i] - 'a'];
        }
        int left = 0;
        for (int right = 0; right < m; ++right) {
            int x = s2[right] - 'a';
            ++cnt[x];
            while (cnt[x] > 0) {
                --cnt[s2[left] - 'a'];
                ++left;
            }
            if (right - left + 1 == n) {
                return true;
            }
        }
        return false;
    }
};
```



## [剑指 Offer II 015. 字符串中的所有变位词](https://leetcode-cn.com/problems/VabMRr/)

**滑动窗口**

```c++
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> vs(26,0);
        vector<int> vp(26,0);
        vector<int> res;

        int ns = s.size(), np = p.size();

        if(ns < np) return res;
        for(int i = 0; i < np; i++){
            vs[s[i] - 'a']++;
            vp[p[i] - 'a']++;
        }
        if(vs == vp) res.push_back(0);

        int l = 0;
        for(int i = np; i < ns; i++){
            vs[s[i] - 'a']++;
            vs[s[l++] - 'a']--;
            if(vs == vp) res.push_back(l);
        }
        return res;
        
    }
};
```



## [剑指 Offer II 016. 不含重复字符的最长子字符串](https://leetcode-cn.com/problems/wtcaE1/)

**滑动窗口**

利用哈希表维护窗口内元素，如果发现当前元素在哈希表中出现了，则移动窗口左边界

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int n = s.size();
        unordered_set<char> hash;
        int i = 0,res = 0;
        for(int j = 0; j < s.size(); j++){
            while(hash.count(s[j])){
                hash.erase(s[i++]);
            }
            hash.insert(s[j]);
            res = max(res, j - i + 1);
        }
        return res;
    }
};
```

## [剑指 Offer II 017. 含有所有字符的最短字符串](https://leetcode-cn.com/problems/M1oyTv/)

滑动窗口，维护字母次数

```c++
class Solution {
public:
    string minWindow(string s, string t) {
        string res;
        unordered_map<char,int> hs;
        unordered_map<char,int> ht;
        for(auto x : t) ht[x]++;
        //用于记录,每在窗口中加入一个所需元素，该数加一，cnt == t.size()时表明窗口内是一个符合条件的最小子串
        int cnt = 0;
        //i , j 分别为窗口的左右两边界
        for(int i = 0, j = 0; j < s.size(); j++){
            //窗口向右移动
            hs[s[j]]++;
            //如果当前元素是t中的字符，
            if(hs[s[j]] <= ht[s[j]]) cnt++;
            //窗口左边界收缩
            while(hs[s[i]] > ht[s[i]]) hs[s[i++]]--;
            //找到一个符合条件的最小字串
            if(cnt == t.size()){
                if(res.empty() || res.size() > j - i + 1)
                    res = s.substr(i,j - i + 1);
            }
        }
        return res;
    }
};
```



## [剑指 Offer II 018. 有效的回文](https://leetcode-cn.com/problems/XltzEq/)

双指针

**介绍几个字符串API**

- `tolower()`:将大写字母转化为小写
- `isalnum`：判断字符是否为字母或数字，也可以自己实现

```c++
class Solution {
public:
    bool check(char s){
        if((s >= 'a' && s <= 'z') || (s >= 'A' && s <= 'Z') || (s >= '0' && s <= '9'))
            return true;
        return false;
    }
    bool isPalindrome(string s) {
        int i = 0, j = s.size() - 1;
        while(i < j){
            while(i < j && !check(s[i])) i++;
            while(i < j && !check(s[j])) j--;
            if(tolower(s[i]) == tolower(s[j])){
                i++;
                j--;
            }else{
                return false;
            }
        }
        return true;
    }
};
```



## [剑指 Offer II 019. 最多删除一个字符得到回文](https://leetcode-cn.com/problems/RQku0D/)

```c++
class Solution {
public:
    bool check(string s, int i, int j){
        while(i < j){
            if(s[i] == s[j]){
                i++;
                j--;
            }else{
                return false;
            }
        }
        return true;
    }
    bool validPalindrome(string s) {
        int i = 0, j = s.size() - 1;
        while(i < j){
            if(s[i] == s[j]){
                i++;
                j--;
            }else{
                return check(s,i+1,j) || check(s,i,j-1);
            }
        }
        return true;
    }
};
```

## [剑指 Offer II 020. 回文子字符串的个数](https://leetcode-cn.com/problems/a7VOhD/)

```c++
class Solution {
public:
    int count(string s, int i, int j){
        if(i > j || j >= s.size()) return 0;
        int l = i, r = j;
        int res = 0;
        while(l >= 0 && r < s.size()){
            if(s[l] == s[r]){
                l--;
                r++;
                res++;
            }else{
                break;
            }
        }
        return res;
    }
    int countSubstrings(string s) {
        int res = 0;
        for(int i = 0; i < s.size(); i++){
            res += count(s,i,i);
            res += count(s,i,i+1);
        }
        return res;
    }
};
```

## [剑指 Offer II 057. 值和下标之差都在给定的范围内](https://leetcode-cn.com/problems/7WqeDu/)

**对于该题，直观想法是对于每一个数，遍历其前面`k`个数，寻找是否存在在区间`[nums[i] - t, nyms[i] + t]`内的数，但是该解法时间复杂度过高，容易超时**

**在暴力解法中其实一直在寻找是否存在落在` [nums[i] - t, nyms[i] + t] `的数，这个过程可以用平衡的二叉搜索树来加速，平衡的二叉树的搜索时间复杂度为` O(logk)`。在 `STL` 中` set` 和 `map` 属于关联容器，其内部由红黑树实现，红黑树是平衡二叉树的一种优化实现，其搜索时间复杂度也为 `O(logk)`。逐次扫码数组，对于每个数字` nums[i]`，当前的 `set `应该由其前 `k `个数字组成，可以` lower_bound `函数可以从` set` 中找到符合大于等于 `nums[i] - t` 的最小的数，若该数存在且小于等于` nums[i] + t`，则找到了符合要求的一对数。**

- 维护一个大小为`k`的滑动窗口，每次寻找窗口内的最接近 `nums[i] - t` 的最小的数，若该数存在且小于等于` nums[i] + t`，则找到了符合要求的一对数。



```C++
class Solution {
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        set<long long> st;
        int left = 0;
        for (int right = 0; right < nums.size(); right ++) {
            if (right - left > k) {
                st.erase(nums[left]);
                left ++;
            }
            auto a = st.lower_bound((long long) nums[right] - t);
            if (a != st.end() && abs(*a - nums[right]) <= t) {
                return true;
            }
            st.insert(nums[right]);
        }
        return false;
    }
};
```

**也可以使用map容器**

```C++
class Solution {
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        map<long long, int> m;
        int left = 0, right = 0;
        while (right < nums.size()) {
            if (right - left > k) {
                m.erase(nums[left]);
                left ++;
            }
            auto a = m.lower_bound((long long) nums[right] - t);
            if (a != m.end() && abs(a->first - nums[right]) <= t) {
                return true;
            }
            m[nums[right]] = right;
            right ++;
        }
        return false;
    }
};

```



# 前缀和

## [剑指 Offer II 010. 和为 k 的子数组](https://leetcode-cn.com/problems/QTMn0o/)

```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int,int> hash;
        int pre = 0, count = 0;
        //一定要注意将 hash[0] 初始化为 1
        //表示当前缀和等于 k 时的情况
        hash[0] = 1;
        for(auto num : nums){
            pre += num;
            if(hash.find(pre - k) != hash.end()){
                count += hash[pre-k];
            }
            hash[pre]++;
        }
        return count;
    }
};
```

## [剑指 Offer II 011. 0 和 1 个数相同的子数组](https://leetcode-cn.com/problems/A1NYOS/)

- 将 0 看作 -1，题目即可化作求**和为0 的子数组的最大长度**‘
- 参考上题，不同的是，上题是求数量，而该题求长度，因此该题中，应利用哈希表存储前缀和对应的下标
  - 由于是求最大长度，我们需要存储的是每个前缀和第一次出现的下标，遇到相同的前缀和不需要更新下标
- 对于子数组长度，若`prefix[j] - prefix[i] == 0`，即`nums[j]`的前缀和等于`nums[i]`的前缀和
  - 此时，数组`nums`中，`nums[i]~nums[j-1]`构成了一个符合题意的数组，即其中0 1 数量相同
  - 子数组长度为 ` j - i`
- 对于`nums[0]`对应的前缀和所对应的下标，应初始化为`-1`,即`hash[0] = -1`

**代码**

```c++
class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        unordered_map<int,int> hash;
        hash[0] = -1;
        int pre = 0, res = 0;
        for(int i = 0; i < nums.size(); i++){
            if(nums[i] == 1) pre++;
            else pre--;

            if(hash.find(pre) != hash.end()){
                int index  = hash[pre];
                res = max(res, i - index);
            }
            else hash[pre] = i;
        }
        return res;
    }
};
```

## [剑指 Offer II 012. 左右两边子数组的和相等](https://leetcode-cn.com/problems/tvdfij/)

- 利用前缀和
  - 当前元素`nums[i]`前缀和为`prefix[i]`，则其右侧元素和为`total - prefix[i] - nums[i]`
  - 左右两边子数组的和相等 即 `prefix[i] = total - prefix[i] - nums[i]`
  - 即 ` 2 * prefix[i] = total - nums[i]`

**代码：**

```c++
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        int n = nums.size();
        int sum = 0;
        int total = 0;
        for(auto num : nums) total += num;
        for(int i = 0; i < n; i++){
            if(2 * sum == total - nums[i]){
                return i;
            }
            sum += nums[i];
        }
        return -1;
    }
};
```

## [303. 区域和检索 - 数组不可变](https://leetcode-cn.com/problems/range-sum-query-immutable/)

**一维数组前缀和**

```c++
class NumArray {
public:
    NumArray(vector<int>& nums) {
        _prefix.resize(nums.size() + 1);
        _prefix[0] = 0;
        for(int i = 1; i <= nums.size(); i++){
            _prefix[i] = _prefix[i-1] + nums[i-1];
        }
    }
    
    int sumRange(int left, int right) {
        if(left > right) return 0;
        if(left < 0 || left >= _prefix.size()) return 0;
        if(right < 0 || right >= _prefix.size()) return 0;
        return _prefix[right + 1] - _prefix[left];
    }
private:
    vector<int> _prefix;
};
```

## [剑指 Offer II 013. 二维子矩阵的和](https://leetcode-cn.com/problems/O4NDxx/)

**可以借鉴[303. 区域和检索 - 数组不可变](https://leetcode-cn.com/problems/range-sum-query-immutable/)，求解时计算每一行的前缀和，然后将前缀和相加**

```c++
class NumMatrix {
public:
    NumMatrix(vector<vector<int>>& matrix){
        int m = matrix.size(), n = matrix[0].size();
        _prefix.resize(m,vector<int>(n+1));
        for(int i = 0; i < m; i++){
            for(int j = 1; j <= n; j++){
                _prefix[i][j] = _prefix[i][j-1] + matrix[i][j-1];
            }
        }
    }
    
    int sumRegion(int row1, int col1, int row2, int col2) {
        if(_prefix.empty() || _prefix[0].empty()) return 0;
        int res = 0;
        for(int i = row1; i <= row2; i++){
            res += _prefix[i][col2 + 1] - _prefix[i][col1];
        }
        return res;
    }
private:
    vector<vector<int>> _prefix;
};
```

### 二维前缀和

- 以`(0,0)`为左上角元素，以`(i,j)`为右下角元素的子矩阵的元素之和称为`(i,j)`的前缀和`pre(i,j)`
- 当`i = 0`或者`j = 0`时，二维前缀和退化为一维前缀和，计算方法不用多说
- 当`i > 0`且`j > 0`时,`pre(i,j) = pre(i-1,j) - pre(i-1,j-1) + pre(i,j-1) - pre(i-1,j-1) + matrix[i][j]`
  - 即`pre(i,j) = pre(i-1,j) + pre(i,j-1) - pre(i-1,j-1) + matrix(i,j)`
- 构造二维前缀数组时要注意：
  - 行与列均比原数组多一
  - `pre(i,j)`表示以元素`matrix(i-1,j-1)`为右下角元素子数组元素和
  - 即`pre(i,j)`表示前`i`行以及前`j`列的所有元素之和
  - 因此初始化时从`(1,1)`开始

```c++
class NumMatrix {
public:
    vector<vector<int>> sums;

    NumMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size();
        if (m > 0) {
            int n = matrix[0].size();
            sums.resize(m + 1, vector<int>(n + 1));
            for (int i = 1; i <= m; i++) {
                for (int j = 1 ;j <=n; j++) {
                    sums[i][j] = sums[i-1][j] + sums[i][j-1] - sums[i-1][j-1] + matrix[i-1][j-1];
                }
            }
        }
    }

    int sumRegion(int row1, int col1, int row2, int col2) {
        return sums[row2 + 1][col2 + 1] - sums[row1][col2 + 1] - sums[row2 + 1][col1] + sums[row1][col1];
    }
} ;
```



# 链表

## [剑指 Offer II 021. 删除链表的倒数第 n 个结点](https://leetcode-cn.com/problems/SLwz0R/)

**使用双指针，令快指针先移动n步，然后两指针同时移动，当快指针指向尾节点时，慢指针正好指向要删除节点的前一个节点**

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(-1,head);
        ListNode* fast = dummy;
        ListNode* slow = dummy;
        while(n--){
            fast = fast->next;
        }
        while(fast->next){
            fast = fast->next;
            slow = slow->next;
        }
        slow->next = slow->next->next;
        return dummy->next;
    }
};
```

## [剑指 Offer II 022. 链表中环的入口节点](https://leetcode-cn.com/problems/c32eOV/)

```c++
class Solution {
public:

    ListNode *detectCycle(ListNode *head) {
        if(!head || !head->next) return nullptr;
        ListNode* slow = head;
        ListNode* fast = head;
        while(fast != NULL && fast->next != NULL){
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow){
                ListNode* index1 = head;
                ListNode* index2 = slow;
                while(index1 != index2){
                    index1 = index1->next;
                    index2 = index2->next;
                }
                return index1;
            }
        }
        return nullptr;
    }
};
```

## [剑指 Offer II 023. 两个链表的第一个重合节点](https://leetcode-cn.com/problems/3u1WK4/)

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA || !headB) return NULL;
        ListNode* p1 = headA;
        ListNode* p2 = headB;
        while(p1 != p2){
            p1 = p1 == NULL ? headB : p1->next;
            p2 = p2 == NULL ? headA : p2->next;
        }
        return p1;
    }
};
```



## [剑指 Offer II 024. 反转链表](https://leetcode-cn.com/problems/UHnkqh/)

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* cur = head;
        ListNode* pre = nullptr;
        ListNode* next;
        while(cur){
            next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
};
```

**递归**

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(head == nullptr || head->next == nullptr){
            return head;
        }else{
            ListNode* cur = reverseList(head->next);
            head->next->next = head;
            head->next = nullptr;
            return cur;
        }
    }
};

```

## [剑指 Offer II 025. 链表中的两数相加](https://leetcode-cn.com/problems/lMSNwu/)

**利用栈进行元素的加法运算，并构建新的结果链表，最后翻转结果链表**

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* cur = head;
        ListNode* pre = nullptr;
        ListNode* next;
        while(cur){
            next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        stack<int> s1;
        stack<int> s2;
        while(l1){
            s1.push(l1->val);
            l1 = l1->next;
        }
        while(l2){
            s2.push(l2->val);
            l2 = l2->next;
        }
        int carry = 0;
        ListNode* root = new ListNode(0);
        ListNode* newHead = root;
        while(!s1.empty() || !s2.empty()){
            int a = s1.empty() ? 0 : s1.top();
            int b = s2.empty() ? 0 : s2.top();
            if(!s1.empty()) s1.pop();
            if(!s2.empty()) s2.pop();
            int sum = a + b + carry;
            int cur = sum % 10;
            carry = sum / 10;
            ListNode* node = new ListNode(cur);
            root->next = node;
            root = root->next;
        }
        if(carry){
            ListNode* node = new ListNode(carry);
            root->next = node;
        }
        
        ListNode* res = reverseList(newHead->next);
        delete newHead;
        return res;
    }
};
```

**发现翻转链表这一步很多余，在处理过程中可以直接令当前节点的next指针指向前一个节点**

处理过程更改如下

```c++
        while(!s1.empty() || !s2.empty()){
            int a = s1.empty() ? 0 : s1.top();
            int b = s2.empty() ? 0 : s2.top();
            if(!s1.empty()) s1.pop();
            if(!s2.empty()) s2.pop();
            int sum = a + b + carry;
            int cur = sum % 10;
            carry = sum / 10;
            ListNode* node = new ListNode(cur,root);
            root = node;
        }
        if(carry){
            ListNode* node = new ListNode(carry,root);
            root = node;
        }
```



## [剑指 Offer II 026. 重排链表](https://leetcode-cn.com/problems/LGjMqU/)

思路对了，代码出错了，首先介绍思路

- 利用快慢指针找出链表中点
- 从中点断开链表并将后半部分链表反转
- 将后半部分链表插入前半链表的缝隙中

**在找中间节点这部分我出错了**

- 我的循环条件是

  ```c++
          while(fast && fast->next){
              fast = fast->next->next;
              slow = slow->next;
          }
  ```

  - 该条件在链表节点数为奇数时是正确的，但为偶数时报错

- 而正确的条件是

  ```c++
          while(fast->next && fast->next->next){
              fast = fast->next->next;
              slow = slow->next;
          }
  ```

  - 此时若节点数为偶数，则`slow`指向前半部分链表的最后一个节点，因此从`slow->next`拆分链表
  - 若节点数为奇数，则`slow`指向中间节点，也从`slow->next`拆分链表

- 还有一种中间节点寻找方式

  ```c++
          while(fast){
              fast = fast->next;
              slow = slow->next;
              if(fast) fast = fast->next;
          }
  ```

  - 此时若节点数为偶数，则`slow`指向后半部分链表的第一个节点，因此从`slow`拆分链表
  - 若节点数为奇数，则`slow`指向中间节点的下一个节点，也从`slow`拆分链表

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* cur = head;
        ListNode* pre = nullptr;
        ListNode* next;
        while(cur){
            next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
    void reorderList(ListNode* head) {
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast->next && fast->next->next){
            fast = fast->next->next;
            slow = slow->next;
        }
        ListNode* head2 = reverseList(slow->next);
        slow->next = nullptr;
        ListNode* head1 = head;
        ListNode* cur = head1;
        while(cur){
            if(cur == head1){
                ListNode* next = head1->next;
                cur->next = head2;
                cur = cur->next;
                head1 = next;
            }else{
                ListNode* next = head2->next;
                cur->next = head1;
                cur = cur->next;
                head2 = next;
            }
        }
    }
};
```



## [剑指 Offer II 027. 回文链表](https://leetcode-cn.com/problems/aMhZSa/)

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head){
        if(!head || !head->next) return head;
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while(cur){
            ListNode* next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }

    bool isPalindrome(ListNode* head) {
        if(!head || !head->next) return true;
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast->next && fast->next->next){
            fast = fast->next->next;
            slow = slow->next;
        }
        ListNode* temp = reverseList(slow->next);
        while(temp){
            if(temp->val == head->val){
                temp = temp->next;
                head = head->next;
            }else{
                return false;
            }
        }
        return true;
    }
};
```



## [剑指 Offer II 028. 展平多级双向链表](https://leetcode-cn.com/problems/Qv1Da2/)

首先碰到该题第一反应就是使用递归

- 确定递归终止条件

  - 遍历到空节点时返回

    ```c++
    if(!head) return head;
    ```

  - 一开始我想的是遍历到空节点或者节点无`next`节点时返回，后来发现不严谨，因为尾节点的`child`指针可能不为空

    ```c++
    if(!head || ！head->next) return head;
    ```

    

- 确定返回值

  - 返回扁平处理完一层后的头节点

- 单层递归逻辑

  - 判断当前节点的`child`指针是否为空，若为空，直接递归处理下一个节点

    ```c++
    if(!head->child) head->next = flatten(head->next);
    ```

  - 如果当前节点存在孩子节点，则要去递归处理孩子层的节点

    - 处理逻辑很简单，首先记录当前节点的下一个节点`next`

    - 然后递归处理孩子节点，记录返回值`cur`，即孩子曾处理完成后的**头节点**

    - 还有记录下处理完成后的孩子层链表的尾部节点`tail`

    - 令当前节点的`next`指向孩子层头节点，孩子曾尾部节点`next`指向当前节点的`next`

    - 千万不能忘了前驱指针的处理

    - 最后一点，对于当前节点的孩子处理完成后，一定要将`child`置空，否则会报错，我在这里卡了很久很久

    - 最后最后一点，由于当前节点可能为尾部节点，因此更新前驱节点时要多做一次判断

      ```c++
      if(next)
      	next->prev = tail;
      ```



**完整代码**

```c++
class Solution {
public:
    Node* flatten(Node* head) {
        if(!head) return head;
        //当前节点的`child`指针是否为空，若为空，直接递归处理下一个节点
        if(!head->child) head->next = flatten(head->next);
        else{
            //记录当前节点的下一个节点
            Node* next = head->next;
            //记录孩子节点，方便置空 child指针
            Node* child = head->child;
            //置空child指针，一定不能遗忘
            head->child = nullptr;
            //递归处理孩子层，返回处理完成后的头节点
            Node* cur = flatten(child);
            //寻找孩子曾处理完成后的链表尾部节点
            Node* tail = cur;
            while(tail->next) tail = tail->next;
            //更新next指针以及前驱指针
            head->next = cur;
            cur->prev = head;
            tail->next = next;
            //next可能为空，因此要判断一下
            if(next)
                next->prev = tail;
        }
        return head;
    }
};
```



## [剑指 Offer II 029. 排序的循环链表](https://leetcode-cn.com/problems/4ueAj6/)

**分类讨论**

**由于是非递减循环链表，因此对于一个节点`cur`,可以通过判断其与其下一个节点值的大小关系来判断是否到了链表的边界**

- 对于链表边界节点`cur`,一定有`cur->val > cur->next->val`，因为最大节点的下一个节点是最小节点
  - 发现了一个问题，按照我的想法，对于边界节点应该是`cur->val >= cur->next->val`，因为它是非递减链表，因此首尾元素有可能相同的
  - 但是这样就报错了，对于示例`[3,3,5] 0`，结果为`[3,0,3,5]`，而正确结果为`[3,3,5,0]`
  - 因为`3 <= 3`,因此错误的判断 第一个`3`就为边界点
  - 因此应该定义为`cur->val > cur->next->val`，对于所有值都相同的极端情况，一定会插入到边界后，会一直循环遍历，直到`cur->next == head`

- 在中间能够找到一个节点`cur`，满足`cur->val<=val<=cur->next->val`，直接插入即可
- 找不到，则一定是所有的值都比它小或大，其实都会插入到边界跳跃点，即找到`cur`，满足`val<=cur->next->val<cur->val`(比最小的还小）或`cur->next->val<cur->val<=val`（比最大的还大）

```c++
class Solution {
public:
    /*
    3种插入情况：
        1) cur.val < x < cur.next.val 顺序插入 
        2) 插入点为为序列的边界跳跃点：
            如 3->4->1 插入5，这样 1(cur->next)<4(cur)<5(x) 4为插入点的前驱节点；这种情况表示x比最大值都大
            如 3->4->1 插入0，这样 0(x)<1(cur->next)<4(cur) 4为插入点的前驱节点；这种情况表示x比最小值都小
    */
    Node* insert(Node* head, int x) {
        if(!head){
            head=new Node(x);
            head->next=head;
            return head;
        }
        Node *cur=head;
        while(cur->next!=head){
            // cur 为边界跳越点
            if(cur->next->val<cur->val){
                if(x>=cur->val)break;// x比最大值都大
                if(x<=cur->next->val)break;// x比最小值都小
            }   
            // 顺序插入x中升序序列中
            if(x>=cur->val&&x<=cur->next->val)break;
            cur=cur->next;
        }
        // 将x插入到cur与cur->next之间
        cur->next=new Node(x,cur->next);
        return head;
    }
};
```

## [剑指 Offer II 031. 最近最少使用缓存](https://leetcode-cn.com/problems/OrIXps/)

**双向链表加哈希表**

```c++
class LRUCache {
public:
    LRUCache(int capacity) : _capacity(capacity){

    }
    
    int get(int key) {
        if(hash.find(key) == hash.end())
            return -1;
        auto node = *hash[key];
        int res = node.second;
        Cache.erase(hash[key]);
        Cache.push_front(node);
        hash[key] = Cache.begin();
        return res;
    }
    
    void put(int key, int value) {
        if(hash.find(key) != hash.end()){
            auto node = *hash[key];
            node.second = value;
            Cache.push_front(node);
            Cache.erase(hash[key]);
            hash[key] = Cache.begin();
        }else{
            if(Cache.size() == _capacity){
                auto back = Cache.back();
                hash.erase(back.first);
                Cache.pop_back();
            }
            pair<int,int> node = {key,value};
            Cache.push_front(node);
            hash[key] = Cache.begin();
        }
    }
private:
    list<pair<int,int>> Cache;
    unordered_map<int,list<pair<int,int>>::iterator> hash;
    int _capacity;
};
```

**自定义双向链表加哈希表**

```c++
struct DlinkedNode{
    int key, value;
    DlinkedNode* pre;
    DlinkedNode* next;
    DlinkedNode(): key(0),value(0),pre(nullptr), next(nullptr){}
    DlinkedNode(int k, int v): key(k), value(v), pre(nullptr), next(nullptr){}
};

class LRUCache {
public:
    LRUCache(int capacity) : _capacity(capacity), _size(0) {
        head = new DlinkedNode();
        tail = new DlinkedNode();
        head->next = tail;
        tail->pre = head;
    }
    
    int get(int key) {
        if(Cache.find(key) != Cache.end()){
            int res = Cache[key]->value;
            auto node = Cache[key];
            moveToHead(node);
            return res;
        }
        return -1;
    }
    
    void put(int key, int value) {
        if(Cache.count(key)){
            Cache[key]->value = value;
            auto node = Cache[key];
            moveToHead(node);
        }else{
            if(_size == _capacity){
                auto node = removeTail();
                Cache.erase(node->key);
                delete node;
                --_size;
            }
            auto node = new DlinkedNode(key, value);
            Cache[key] = node;
            addToHead(node);
            ++_size;
        }
    }

    void addToHead(DlinkedNode* node){
        DlinkedNode* next = head->next;
        head->next = node;
        node->pre = head;
        node->next = next;
        next->pre = node;
    }

    void removeNode(DlinkedNode* node){
        node->pre->next = node->next;
        node->next->pre = node->pre;
    }

    void moveToHead(DlinkedNode* node){
        removeNode(node);
        addToHead(node);
    }

    DlinkedNode* removeTail(){
        DlinkedNode* node = tail->pre;
        removeNode(node);
        return node;
    }
private:
    int _capacity;
    int _size;
    DlinkedNode* head;
    DlinkedNode* tail;
    unordered_map<int,DlinkedNode*> Cache;
};
```



# 哈希表

## [剑指 Offer II 030. 插入、删除和随机访问都是 O(1) 的容器](https://leetcode-cn.com/problems/FortPu/)

**哈希表加数组**

- 在数组中进行删除操作可以先将要删除元素移到数组尾部，然后进行删除

```c++
class RandomizedSet {
public:
    /** Initialize your data structure here. */
    RandomizedSet() {
        //随机数发生器初始化
        //srand和rand()配合使用产生伪随机数序列
        srand((unsigned)time(NULL));
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    bool insert(int val) {
        if(hash.find(val) != hash.end())
            return false;
        nums.push_back(val);
        hash[val] = nums.size()-1;
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    bool remove(int val) {
        if(hash.find(val) == hash.end())
            return false;
        int back = nums.back();
        int index = hash[val];
        nums[index] = back;
        nums.pop_back();
        hash[back] = index;
        hash.erase(val);
        return true;
    }
    
    /** Get a random element from the set. */
    int getRandom() {
        int index = rand()%nums.size();
        return nums[index];
    }
private:
    vector<int> nums;
    unordered_map<int,int> hash;
};


```



## [剑指 Offer II 032. 有效的变位词](https://leetcode-cn.com/problems/dKk3P7/)

**当给定字符串中只包含小写字母时可以用数组代替哈希表**

- 要注意该题中当两字符串相同时不属于字母异位词

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        vector<int> vs(26,0);
        vector<int> vt(26,0);
        if(s == t) return false;
        if(s.size() != t.size()) return false;
        for(int i = 0; i < s.size(); i++)
            vs[s[i] - 'a']++;
        for(int i = 0; i < t.size(); i++)
            vt[t[i] - 'a']++;
        
        return vs == vt;
    }
};
```

**输入字符串中包含unicode字符时，利用哈希表**

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        if(s.size() != t.size()) return false;
        if(s == t) return false;
        unordered_map<char,int> hash;
        for(int i = 0; i < s.size(); i++){
            hash[s[i]]++;
        }
        for(int i = 0; i < t.size(); i++){
            if(hash.find(t[i]) == hash.end())
                return false;
            hash[t[i]]--;
            if(hash[t[i]] < 0) return false;
        }
        return true;
    }
};
```

## [剑指 Offer II 033. 变位词组](https://leetcode-cn.com/problems/sfvd7V/)

**变位词按字母序排序后结果是相同**

- 利用哈希表，键值为变位词排序后的结果，`value`为字符串数组
  - 如此，互为变位词的单词的键值是相同，会加入同一个字符串数组
- 遍历数组，对每一个单词排序计算键值，然后根据键值加入哈希表
- 最后只需遍历哈希表，将字符串数组依次加入结果集

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string,vector<string>>hash;
        for(auto str : strs){
            string key = str;
            sort(key.begin(),key.end());
            hash[key].push_back(str);
        }
        vector<vector<string>> res;
        for(auto x : hash){
            res.push_back(x.second);
        }
        return res;
    }
};
```

## [剑指 Offer II 034. 外星语言是否排序](https://leetcode-cn.com/problems/lwyVBB/)

**依次判断数组内相邻单词的大小即可，因为`a<=b && b<= c`则有`a <= c`**

- **因为只包含小写字母，因此可以使用26的数组代替哈希表，将字母表的顺序映射**

  ```c++
  vector<int> hash(26,0);
  for(int i = 0; i < order.size(); i++){
      hash[order[i]-'a'] = i;
  }
  ```

- 对相邻单词每个字母一一比较

  ```c++
  for(int i = 0; i < words.size()-1; i++){
      for(int j = 0,k = 0; j < words[i].size() && k < words[i+1].size(); j++, k++){
          //对于两单词，对应的字母的字母序不符要求
          if(hash[words[i][j] - 'a'] > hash[words[i+1][k] - 'a'])
              return false;
          else if(hash[words[i][j] - 'a'] < hash[words[i+1][k] - 'a'])
              break;
          else
              //后面一个单词遍历完成后，前一个单词还有多的字母，不服字母序
              if(k == words[i+1].size() - 1 && j < words[i].size() - 1)
                  return false;
      }
  }
  ```

  

```c++
class Solution {
public:
    bool isAlienSorted(vector<string>& words, string order) {
        vector<int> hash(26,0);
        for(int i = 0; i < 26; i++){
            hash[order[i]-'a'] = i;
        }
        for(int i = 0; i < words.size()-1; i++){
            for(int j = 0,k = 0; j < words[i].size() && k < words[i+1].size(); j++, k++){
                //对于两单词，对应的字母的字母序不符要求
                if(hash[words[i][j] - 'a'] > hash[words[i+1][k] - 'a'])
                    return false;
                else if(hash[words[i][j] - 'a'] < hash[words[i+1][k] - 'a'])
                    break;
                else
                    //后面一个单词遍历完成后，前一个单词还有多的字母，不服字母序
                    if(k == words[i+1].size() - 1 && j < words[i].size()-1)
                        return false;
            }
        }
        return true;
    }
};
```



# [剑指 Offer II 035. 最小时间差](https://leetcode-cn.com/problems/569nqc/)

**首先对数组排序，然后最小时间差一定出现的相邻元素之间或者首尾元素之间**

```c++
class Solution {
public:
    int getMinutes(string &t) {
        return (int(t[0] - '0') * 10 + int(t[1] - '0')) * 60 + int(t[3] - '0') * 10 + int(t[4] - '0');
    }

    int findMinDifference(vector<string>& timePoints) {
        int res = INT_MAX;
        sort(timePoints.begin(), timePoints.end());
        int pre = getMinutes(timePoints[0]);
        for(int i = 1; i < timePoints.size(); i++){
            int cur = getMinutes(timePoints[i]);
            res = min(cur - pre, res);
            pre = cur;
        }
        //计算首尾元素时间差
        res = min(res, getMinutes(timePoints[0]) - pre + 1440);
        return res;
    }
};
```

**鸽巢原理**

- 一天一共有 `24 * 60 = 1440`种时间，因此当给定数组长度超过`1440`时，则说明出现了重复时间，那么最小时间差必定为0

- 因此可加上判断

  ```c++
  if(timePoints.size() > 1440) return 0;
  ```

  

**也叫抽屉原理，即将5双袜子放入4个抽屉，则必定有一个抽屉装了两双袜子**

```c++
class Solution {
public:
    int getMinutes(string &t) {
        return (int(t[0] - '0') * 10 + int(t[1] - '0')) * 60 + int(t[3] - '0') * 10 + int(t[4] - '0');
    }

    int findMinDifference(vector<string>& timePoints) {
        if(timePoints.size() > 1440) return 0;
        int res = INT_MAX;
        sort(timePoints.begin(), timePoints.end());
        int pre = getMinutes(timePoints[0]);
        for(int i = 1; i < timePoints.size(); i++){
            int cur = getMinutes(timePoints[i]);
            res = min(cur - pre, res);
            pre = cur;
        }
        //计算首尾元素时间差
        res = min(res, getMinutes(timePoints[0]) - pre + 1440);
        return res;
    }
};
```



# 栈 & 单调栈

## [剑指 Offer II 036. 后缀表达式](https://leetcode-cn.com/problems/8Zf90G/)

逆波兰表达式严格遵循「从左到右」的运算。计算逆波兰表达式的值时，使用一个栈存储操作数，从左到右遍历逆波兰表达式，进行如下操作：

- 如果遇到操作数，则将操作数入栈；

- 如果遇到运算符，则将两个操作数出栈，其中先出栈的是右操作数，后出栈的是左操作数，使用运算符对两个操作数进行运算，将运算得到的新操作数入栈。


整个逆波兰表达式遍历完毕之后，栈内只有一个元素，该元素即为逆波兰表达式的值。

```c++
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> s;
        for(int i = 0; i < tokens.size(); i++){
            if(tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/"){
                int num1 = s.top();
                s.pop();
                int num2 = s.top();
                s.pop();
                if (tokens[i] == "+") s.push(num2 + num1);
                if (tokens[i] == "-") s.push(num2 - num1);
                if (tokens[i] == "*") s.push(num2 * num1);
                if (tokens[i] == "/") s.push(num2 / num1);
            }else{
                s.push(stoi(tokens[i]));
            }
        }
        return s.top();
    }
};
```



## [剑指 Offer II 037. 小行星碰撞](https://leetcode-cn.com/problems/XagZNi/)

**首先弄清楚有几种情况**

1. 当前行星`nums[i]`向右走
   1. 若`nums[i-1]`向左走，两行星互不影响
   2. 若`nums[i-1]`向右走，两行星依旧互不影响
2. 当前行星`nums[i]`向左走
   1. 若`nums[i-1]`向左走，两行星互不影响
   2. 若`nums[i-1]`向右走，则两行星发生碰撞

**可以利用栈的思想**

- 若当前行星`nums[i]`向右走，即`nums[i] > 0`,则直接入栈
- 若当前行星`nums[i]`向左走，即`nums[i] < 0`,则需分情况讨论
  - 栈顶元素即为当前行星前一个元素
  - 若栈顶元素大于0，则发生碰撞，对碰撞后的结果继续进行上述判断
  - 若栈顶元素小于0或者栈为空，则直接入栈

```c++
class Solution {
public:
    vector<int> asteroidCollision(vector<int>& asteroids) {
        stack<int> s;
        vector<int> ans;
        for(int i = 0; i < asteroids.size(); i++){
            //行星向右走
            if(asteroids[i] > 0){
                s.push(asteroids[i]);
            }else{
                //行星向左走
                //栈为空或者栈顶元素向左走，两者不碰撞
                if(s.empty() || s.top() < 0) s.push(asteroids[i]);
                //栈顶行星向右走
                else{
                    int cur = asteroids[i];
                    while(cur < 0 && !s.empty() && s.top() > 0){
                        //当前行星与栈顶行星大小相同，则两者一起撞毁
                        if(abs(cur) == s.top()){
                            s.pop();
                            cur = 0;
                            break;
                        //当前行星小于栈顶行星,当前行星销毁
                        }else if(abs(cur) < s.top()){
                            cur = 0;
                            break;
                        //当前行星大于栈顶行星，栈顶行星销毁，即将栈顶元素出栈，继续与下一个栈顶元素比较判断
                        }else{
                            s.pop();
                        }
                    }
                    //若碰撞最后当前行星还在，则入栈，此时栈内元素一定是碰撞完成后的结果
                    if(cur != 0) s.push(cur);
                }
            }
        }
        if(s.empty()) return ans;
        ans.resize(s.size());
        for(int i = ans.size() - 1; i >= 0; i--){
            ans[i] = s.top();
            s.pop();
        }
        return ans;
    }
};
```

## [剑指 Offer II 038. 每日温度](https://leetcode-cn.com/problems/iIQa4I/)

```c++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        stack<int> s;
        int n = temperatures.size();
        vector<int> res(n,0);
        for(int i = 0; i < n ; i++){
            while(!s.empty() && temperatures[s.top()] < temperatures[i]){
                int cur = s.top();
                s.pop();
                res[cur] = i - cur;
            }
            s.push(i);
        }
        return res;
    }
};
```

## [剑指 Offer II 039. 直方图最大矩形面积](https://leetcode-cn.com/problems/0ynMMM/)

```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        //递增栈
        stack<int> s;
        heights.push_back(0);
        int res = 0;
        for(int i = 0; i < heights.size(); i++){
            while(!s.empty() && heights[i] < heights[s.top()]){
                int h = heights[s.top()];
                s.pop();
                //底边长应该用右边第一个比它矮的元素下标 - 左边第一个比他矮的下标 - 1
                //若栈为空，即当前元素为最矮的元素，则底边长应该等于 i
                int w = i;
                if(!s.empty())
                    w = i - s.top() - 1;
                res = max(res, h * w);
            }
            s.push(i);
        }
        return res;
    }
};
```



## [剑指 Offer II 040. 矩阵中最大的矩形](https://leetcode-cn.com/problems/PLYXKQ/)

**转化为直方图中最大矩形面积**

```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        //递增栈
        stack<int> s;
        heights.push_back(0);
        int res = 0;
        for(int i = 0; i < heights.size(); i++){
            while(!s.empty() && heights[i] < heights[s.top()]){
                int h = heights[s.top()];
                s.pop();
                //底边长应该用右边第一个比它矮的元素下标 - 左边第一个比他矮的下标 - 1
                //若栈为空，即当前元素为最矮的元素，则底边长应该等于 i
                int w = i;
                if(!s.empty())
                    w = i - s.top() - 1;
                res = max(res, h * w);
            }
            s.push(i);
        }
        return res;
    }
    int maximalRectangle(vector<string>& matrix) {
        int m = matrix.size();
        if(!m) return 0;
        int n = matrix[0].size();
        if(!n) return 0;
        int res = 0;
        vector<vector<int>> _matrix(m,vector<int>(n,0));
        for(int i = 0; i < n; i++)
            _matrix[0][i] = matrix[0][i] - '0';

        for(int i = 1; i < m; i++){
            for(int j = 0; j < n; j++){
                if(matrix[i][j] == '1'){
                    _matrix[i][j] = _matrix[i-1][j] + 1;
                }
            }
        }
        for(auto x : _matrix){
            res = max(res, largestRectangleArea(x));
        }
        return res;
    }
};
```



# 队列

## [剑指 Offer II 041. 滑动窗口的平均值](https://leetcode-cn.com/problems/qIsx9U/)

```c++
class MovingAverage {
private:
    int len = 0;
    queue<int> nums;
    double sum = 0;
public:
    MovingAverage(int size) {
        len = size;
    }
    
    double next(int val) {
        nums.push(val);
        sum += val;
        if (nums.size() > len) {
            sum -= nums.front();
            nums.pop();
        }
        return sum / nums.size();
    }
};
```

## [剑指 Offer II 042. 最近请求次数](https://leetcode-cn.com/problems/H8086Q/)

**将题意转化一下，3000ms就代表滑动窗口大小为3000，当新加入元素与队列头部元素相差大于3000时，头部元素需要弹出队列，因此队列的元素均是以队列尾部元素往前3000ms内的数据，队列长度即为在该时间段内操作次数**

```c++
class RecentCounter {
public:
    RecentCounter() {

    }
    
    int ping(int t) {
        q.push(t);
        while(t - q.front() > 3000) q.pop();
        return q.size();
    }
private:
    queue<int> q;
};
```



## [剑指 Offer II 043. 往完全二叉树添加节点](https://leetcode-cn.com/problems/NaqhDT/)

**双向队列**

- 对完全二叉树进行层序遍历
- 层序遍历是从上至下，从左至右进行的，每遍历到一个左或右子节点为空的节点时，将其加入双向队列尾部
- 进行插入操作时，从双向队列首部取出一个元素，判断是将其加在左节点还是右节点
  - 若加在右节点，则当前节点左右两子节点加满，将其弹出双向队列
- 插入完成后，将新节点加入双向队列尾部

```c++
class CBTInserter {
public:
    CBTInserter(TreeNode* root):_root(root) {
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            TreeNode* cur = q.front();
            if(!cur->left || !cur->right)
                dq.push_back(cur);
            if(cur->left)
                q.push(cur->left);
            if(cur->right)
                q.push(cur->right);

            q.pop();
        }
    }
    
    int insert(int val) {
        TreeNode* node = dq.front();
        TreeNode* cur = new TreeNode(val);
        if(!node->left)
            node->left = cur;
        else{
            node->right = cur;
            dq.pop_front();
        }
        dq.push_back(cur);
        return node->val;
    }
    
    TreeNode* get_root() {
        return _root;
    }
private:
    TreeNode* _root ;
    deque<TreeNode*> dq;
};
```



## [剑指 Offer II 044. 二叉树每层的最大值](https://leetcode-cn.com/problems/hPov7L/)

```c++
class Solution {
public:
    vector<int> largestValues(TreeNode* root) {
        vector<int> res;
        queue<TreeNode*> q;
        if(!root) return res;
        q.push(root);
        while(!q.empty()){
            int n = q.size();
            int mmax = INT_MIN;
            for(int i = 0; i < n; i++){
                TreeNode* cur = q.front();
                q.pop();
                mmax = max(mmax, cur->val);
                if(cur->left)
                    q.push(cur->left);
                if(cur->right)
                    q.push(cur->right); 
            }
            res.push_back(mmax);
        }
        return res;
    }
};
```

## [剑指 Offer II 045. 二叉树最底层最左边的值](https://leetcode-cn.com/problems/LwUNpT/)

```c++
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        queue<TreeNode*> q;
        q.push(root);
        int res = 0;
        while(!q.empty()){
            int n = q.size();
            for(int i = 0; i < n; i++){
                TreeNode* cur = q.front();
                q.pop();
                if(i == 0) res = cur->val;
                if(cur->left)
                    q.push(cur->left);
                if(cur->right)
                    q.push(cur->right);
            }
        }
        return res;
    }
};
```



## [剑指 Offer II 046. 二叉树的右侧视图](https://leetcode-cn.com/problems/WNC0Lk/)

```c++
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        queue<TreeNode*> q;
        if(!root) return vector<int>{};
        q.push(root);
        vector<int> res;
        while(!q.empty()){
            int n = q.size();
            int ans = 0;
            for(int i = 0; i < n; i++){
                TreeNode* cur = q.front();
                q.pop();
                if(i == n-1) ans = cur->val;
                if(cur->left)
                    q.push(cur->left);
                if(cur->right)
                    q.push(cur->right);
            }
            res.push_back(ans);
        }
        return res;
    }
};
```

**递归解法**

```c++
class Solution {
public:
    vector<int> res;
    void dfs(TreeNode* root, int depth){
        if(!root) return;
        if(res.size() == depth){
            res.push_back(root->val);
        }
        dfs(root->right, depth + 1);
        dfs(root->left, depth + 1);
    }
    vector<int> rightSideView(TreeNode* root) {
        dfs(root,0);
        return res;
    }   
};
```



# 二叉树

## [剑指 Offer II 047. 二叉树剪枝](https://leetcode-cn.com/problems/pOCWxh/)

**采用后序遍历，先处理左右子节点，再处理根节点**

- 当前节点值为`0`并且其为叶子节点时，则需删除
  - 我卡在了删除操作这里，因为函数传入普通指针时，将其置空，并不会影响本来的指针
    - 传入的是指针变量，但实际上是值传递，因为实际上传入的值是一个地址，传入后形成了一个与实参相同但独立的指针
    - 这样可以更改指针指向区域的值，但无法更改指针本身的指向
  - 因此使用传入引用的方式

```c++
class Solution {
public:
    void dfs(TreeNode* &root){
        if(!root) return ;
        dfs(root->left);
        dfs(root->right);
        if(!root->left && !root->right && root->val == 0){
            root = nullptr;
        }
    }
    TreeNode* pruneTree(TreeNode* root) {
        dfs(root);
        return root;
    }
};
```

**另一种后序方式**

```c++
class Solution {
public:
    TreeNode* pruneTree(TreeNode* root) {
        if (root == nullptr) {
            return nullptr;
        }
        TreeNode* left = pruneTree(root->left);
        TreeNode* right = pruneTree(root->right);
        if (root->val == 0 && left == nullptr && right == nullptr) {
            return nullptr;
        }
        root->left = left;
        root->right = right;
        return root;
    }
};
```

## [剑指 Offer II 048. 序列化与反序列化二叉树](https://leetcode-cn.com/problems/h54YBf/)

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
       string res;
       if(!root) return "";
       queue<TreeNode*> q;
       q.push(root);
       while(!q.empty()){
            TreeNode* cur = q.front();
            q.pop();
            if(cur)
                res += to_string(cur->val);
            else{
                res += '#';
            }
            res += ',';
            if(cur){
                q.push(cur->left);
                q.push(cur->right);
            }
       }
       return res;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string s) {
        vector<TreeNode*> v;
        if(s.empty()) return nullptr;
		int j = 0;
        while(j < s.size())
        {
            string stmp = "";
            while(s[j] != ',')
            {
                stmp += s[j];
                j++;
            }

            if(stmp == "#")
            {
                v.push_back(nullptr);
            }
            else
            {
                int tmp = atoi(stmp.c_str());
                TreeNode* newnode = new TreeNode(tmp);
                v.push_back(newnode);
            }
            j++;
        }
        int p = 1;
        for(int i = 0; i < v.size(); i++){
            if(v[i]){
                v[i]->left = v[p++];
                v[i]->right = v[p++];
            }
        }
        return v[0];
    }
};

// Your Codec object will be instantiated and called as such:
// Codec ser, deser;
// TreeNode* ans = deser.deserialize(ser.serialize(root));
```

## [剑指 Offer II 049. 从根节点到叶节点的路径数字之和](https://leetcode-cn.com/problems/3Etpl5/)

```c++
class Solution {
public:
    int res;
    void dfs(TreeNode* root,int path){
        if(!root) return;
        //计算路径值
        path = path * 10 + root->val;
        //遍历到根节点，加入结果
        if(!root->left && !root->right){
            res += path;
        }

        dfs(root->left, path);
        dfs(root->right, path);
    }
    int sumNumbers(TreeNode* root) {
        dfs(root,0);
        return res;
    }
};
```

## [剑指 Offer II 050. 向下的路径节点之和](https://leetcode-cn.com/problems/6eUYwP/)

**双重递归**

```c++
class Solution {
public:
    int dfs(TreeNode* root, int targetSum, int sum){
        if(!root) return 0;
        int res = 0;
        sum += root->val;
        if(sum == targetSum)
            res++;
        res += dfs(root->left,targetSum,sum);
        res += dfs(root->right,targetSum,sum);
        return res;
    }

    int pathSum(TreeNode* root, int targetSum) {
        if(!root) return 0;
        int res = 0;
        res += dfs(root,targetSum,0);
        res += pathSum(root->left,targetSum);
        res += pathSum(root->right,targetSum);
        return res;
    }
};
```

**前缀和**

```c++
class Solution {
    unordered_map<long long, int> hash;
public:
    int dfs(TreeNode* root, int targetSum, long long sum){
        if(!root) return 0;
        int res = 0;
        sum += root->val;
        if(hash.count(sum - targetSum)){
            res += hash[sum - targetSum];
        }

        hash[sum]++;
        res += dfs(root->left,targetSum,sum);
        res += dfs(root->right,targetSum,sum);
        hash[sum]--;
        return res;
    }

    int pathSum(TreeNode* root, int targetSum) {
        if(!root) return 0;
        hash[0] = 1;
        int res = dfs(root,targetSum,0);

        return res;
    }
};
```

## [剑指 Offer II 051. 节点之和最大的路径](https://leetcode-cn.com/problems/jC7MId/)

```c++
class Solution {
public:
    int res = INT_MIN;
    int dfs(TreeNode* root){
        if(!root) return 0;
		//这里出错了，当为负数时，可以选择不加上该节点，即令其等于 0
        int left = max(0,dfs(root->left));
        int right = max(0,dfs(root->right));

        int midValue = left + right + root->val;
        res = max(res, midValue);

        int val = root->val + max(left,right);
        return val;

    }
    int maxPathSum(TreeNode* root) {
        dfs(root);
        return res;
    }
};
```

## [剑指 Offer II 052. 展平二叉搜索树](https://leetcode-cn.com/problems/NYBBNL/)

```c++
class Solution {
public:
    TreeNode* pre = nullptr;
    TreeNode* head = nullptr;
    void dfs(TreeNode* root)
    {
        if(!root) return;
        dfs(root->left);
        
        if(pre){
            pre->right = root;
        }else
            head = root;

        pre = root;

        root->left = nullptr;

        dfs(root->right);
    }
    TreeNode* increasingBST(TreeNode* root) {
        dfs(root);
        return head;
    }
};
```

## [剑指 Offer II 053. 二叉搜索树中的中序后继](https://leetcode-cn.com/problems/P5rCT8/)

```c++
class Solution {
public:
    TreeNode* pre = nullptr;
    TreeNode* res;
    void dfs(TreeNode* root, TreeNode* p){
        if(!root) return;
        dfs(root->left,p);

        if(pre == p) res = root;
        pre = root;

        dfs(root->right,p);
    }
    TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
        dfs(root,p);
        return res;
    }
};
```

## [剑指 Offer II 054. 所有大于等于节点的值之和](https://leetcode-cn.com/problems/w6cpku/)

```c++
class Solution {
public:
    //遍历顺序右 根 左
    int pre = 0;
    void dfs(TreeNode* root){
        if(!root) return;
        dfs(root->right);

        root->val += pre;
        pre = root->val;

        dfs(root->left);
    }
    TreeNode* convertBST(TreeNode* root) {
        dfs(root);
        return root;
    }
};
```

## [剑指 Offer II 055. 二叉搜索树迭代器](https://leetcode-cn.com/problems/kTOapQ/)

**递归**

```c++
class BSTIterator {
public:
    BSTIterator(TreeNode* root) {
        _cur = -1;
        dfs(root);
    }

    void dfs(TreeNode* root){
        if(!root) return;
        dfs(root->left);
        _inorder.push_back(root);
        dfs(root->right);
    }

    
    int next() {
        if(_cur == -1){
            _cur++;
            return _inorder[0]->val;
        }
        return _inorder[++_cur]->val;
    }
    
    bool hasNext() {
        if(_cur == -1) return _inorder.size();
        return _cur < _inorder.size() - 1;
    }
private:
    vector<TreeNode*> _inorder;
    int _cur;
};
```

**迭代**

```c++
class BSTIterator {
private:
    TreeNode* cur;
    stack<TreeNode*> stk;
public:
    BSTIterator(TreeNode* root): cur(root) {
        stk.push(cur);
    }
    
    int next() {
        while(!stk.empty()){
            TreeNode* node = stk.top();
            if(node){
                stk.pop();
                if(node->right) stk.push(node->right);  //右
                
                stk.push(node);                         //中
                stk.push(nullptr);
                
                if(node->left) stk.push(node->left);    //左
            }else{
                stk.pop();
                TreeNode* ret = stk.top();
                stk.pop();
                return ret->val;
            }
        }
        return -1;
    }
    
    bool hasNext() {
        return !stk.empty();
    }
};
```



## [剑指 Offer II 056. 二叉搜索树中两个节点之和](https://leetcode-cn.com/problems/opLdQZ/)

```c++
class Solution {
public:
    unordered_set<int> hashTable;

    bool findTarget(TreeNode *root, int k) {
        if (root == nullptr) {
            return false;
        }
        if (hashTable.count(k - root->val)) {
            return true;
        }
        hashTable.insert(root->val);
        return findTarget(root->left, k) || findTarget(root->right, k);
    }
};

```



# 桶排序

## [剑指 Offer II 057. 值和下标之差都在给定的范围内](https://leetcode-cn.com/problems/7WqeDu/)

因为题目只关心的是差的绝对值小于等于 `t `的数字，这时候容易联想到桶，可以把数字放进若干个大小为` t + 1 `的桶内，这样的好处是一个桶内的数字绝对值差肯定小于等于` t`。对于桶的标号进行说明，例如` [0, t]` 放进编号为` 0` 的桶内，`[t + 1, 2t + 1] `放进编号为 1 的桶内，对于负数，则 `[-t - 1, -1] `放进编号为` -1 `的桶内，`[-2t - 2, -t - 2] `编号为` -2 `的桶内，可以发现桶

- `n >= 0 : ID = n / (t + 1)`
- `n < 0 : ID = (n + 1) / (t + 1) - 1`

**算法流程：**

**桶的大小为`t+1`，则属于同一个桶的两元素必定满足绝对值差小于等于` t`**

**桶的数量为`k`，即只考虑当前数字的与前`k`个数字的关系，即在遍历过程中不断删除前面超过下标界限的元素所在的桶**

- 对于当前元素，计算其所属桶，若桶中已有元素，则返回`true`
- 否则去其左右相邻的桶寻找
  - 若存在左右相邻桶，则判断里面的元素是否满足绝对值差小于等于` t`，若满足，则返回`true`
- 记录当前元素桶信息
- 移除超出下标界限的桶信息

```c++
class Solution {
public:
    int getID(int n, long size) {
        return (n >= 0) ? n / size : (n + 1) / size - 1;
    }

    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        unordered_map<int, int> mp;
        long bucketSize = static_cast<long>(t) + 1;
        for (int i = 0; i < n; i++) {
            long x = nums[i];
            int id = getID(x, bucketSize);
            if (mp.count(id)) {
                return true;
            }
            if (mp.count(id - 1) && abs(x - mp[id - 1]) <= t) {
                return true;
            }
            if (mp.count(id + 1) && abs(x - mp[id + 1]) <= t) {
                return true;
            }
            mp[id] = x;
            if (i >= k) {
                mp.erase(getID(nums[i - k], bucketSize));
            }
        }
        return false;
    }
};
```

