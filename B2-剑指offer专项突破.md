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

**不使用 Long 类型解法**

```c++
class Solution {
public:
    int divide(int a, int b) {
        if (a == INT_MIN && b == -1) {
            return INT_MAX;
        }
        int negative = 2;
        if (a > 0) {
            negative--;
            a = -a;
        }
        if (b > 0) {
            negative--;
            b = -b;
        }
        unsigned int ret = divideCore(a, b);
        return negative == 1 ? -ret : ret;
    }

    unsigned int divideCore(int a, int b) {
        int ret = 0;
        // 注意a, b都是负数，所以a <= b就是还可以继续除
        while (a <= b) {
            int value = b;
            unsigned int quo = 1;
            while (value >= 0xc0000000 && a <= value + value) {
                quo *= 2;
                value *= 2;
            }
            ret += quo;
            a -= value;
        }
        return ret;
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
  - 则此时子数组长度为`l - i + 1`

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
            /*
            	sum[l] - sum[i-1] >= target
				即nums[i]~nums[l]的元素和 >= target
            */
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
  - 但是实现的时候卡在了判断字符是否属于另一字符串上
- **题解中，窗口用于维护字符数量，因为根据题意，当满足条件时，窗口内元素个数与另一字符串元素个数是一致的**，妙哉
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
    //前缀和数组，sum[i][j]表示以[0,0]为左上角元素，[i-1,j-1]为右下角元素的子数组所有元素之和
    //sum[0][j]以及sum[i][0]全初始化为0
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

**栈**

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        stack<ListNode*> s;
        ListNode* dummy = new ListNode(-1,head);
        ListNode* cur = dummy;
        while(cur){
            s.push(cur);
            cur = cur->next;
        }

        while(n--){
            s.pop();
        }

        ListNode* pre = s.top();
        pre->next = pre->next->next;
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
                //如果两者都为空，则说明不存在环，直接返回
                if(!fast) return nullptr;
                ListNode* index1 = head;
                ListNode* index2 = fast;
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

    - 然后递归处理孩子节点，记录返回值`cur`，即孩子层处理完成后的**头节点**

    - 还有记录下处理完成后的孩子层链表的尾部节点`tail`

    - 令当前节点的`next`指向孩子层头节点，孩子层尾部节点`next`指向当前节点的`next`

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

**二刷记录**

```c++
class Solution {
public:
    Node* insert(Node* head, int insertVal) {
        Node* node = new Node(insertVal);
        if(!head){
            node->next = node;
            head = node;
            return head;
        }
        
        Node* cur = head;
        while(cur->next != head){
            if(cur->val > cur->next->val){
                if(cur->val <= insertVal || cur->next->val >= insertVal){
                    Node* next = cur->next;
                    cur->next = node;
                    node->next = next;
                    return head;
                }
            }
            if(cur->val <= insertVal && insertVal <= cur->next->val){
                    Node* next = cur->next;
                    cur->next = node;
                    node->next = next;
                    return head;
            }
            cur = cur->next;
        }
        //如以上条件均不满足，直接将点插入头节点后面，这一步非常容易忘记
        Node* next = cur->next;
        cur->next = node;
        node->next = next;
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

**二刷：**

```c++
class Solution {
public:
    TreeNode* pruneTree(TreeNode* root) {
        if(!root) return nullptr;
        root->left = pruneTree(root->left);
        root->right = pruneTree(root->right);
        if(root->val == 0 && !root->left && !root->right){
            root = nullptr;
        }
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

**二刷**

```c++
class Solution {
public:
    unordered_map<int,int> mp;
    int res;
    void dfs(TreeNode* root, int targetSum, int path){
        if(!root) return ;
        path += root->val;

        if(mp.count(path - targetSum)){
            res += mp[path - targetSum];
        }
        mp[path]++;

        dfs(root->left, targetSum, path);
        dfs(root->right, targetSum, path);

        mp[path]--;
    }
    int pathSum(TreeNode* root, int targetSum) {
        mp[0]++;
        dfs(root,targetSum,0);
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



# 数据结构

## map&set

`map`和`set`底层为红黑树，是有序容器,且支持快速插入

之前从未接触过`map`容器里的两个成员函数

- `map.lower_bound(k)`：指向键大于等于k的第一个元素，注意返回值是一个迭代器，即指针
- `map.upper_bound(k)`：指向键大于k的第一个元素

```c++
map_name.lower_bound(key)
```

该函数返回指向容器中键的迭代器，该迭代器等效于参数中传递的k。

## 优先队列`priority_queue<int,vector<int>,greater<int>>`

## [剑指 Offer II 057. 值和下标之差都在给定的范围内](https://leetcode-cn.com/problems/7WqeDu/)

**对于该题，直观想法是对于每一个数，遍历其前面`k`个数，寻找是否存在在区间`[nums[i] - t, nums[i] + t]`内的数，但是该解法时间复杂度过高，容易超时**

**在暴力解法中其实一直在寻找是否存在落在` [nums[i] - t, nums[i] + t] `的数，这个过程可以用平衡的二叉搜索树来加速，平衡的二叉树的搜索时间复杂度为` O(logk)`。在 `STL` 中` set` 和 `map` 属于关联容器，其内部由红黑树实现，红黑树是平衡二叉树的一种优化实现，其搜索时间复杂度也为 `O(logk)`。逐次扫码数组，对于每个数字` nums[i]`，当前的 `set `应该由其前 `k `个数字组成，可以` lower_bound `函数可以从` set` 中找到符合大于等于 `nums[i] - t` 的最小的数，若该数存在且小于等于` nums[i] + t`，则找到了符合要求的一对数。**

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



**map + 桶排序**

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

## [剑指 Offer II 058. 日程表](https://leetcode-cn.com/problems/fi9suh/)

对于该题，暴力法就是对于给定`(start,end)`，遍历所有已经存放的日期`(i,j)`，检查是否有重复的部分

- 当`start >= j || end <= i`时，区间不重合

暴力法事件复杂度为`O(N^2)`

如果我们在有序集合中使用二分查找来检查新日程是否可以插入，则时间复杂度为`O(logN)`

因此，需要一个支持快速插入并有序的数据结构，在`c++`中，`map`底层为红黑树，里面的元素有序存放，并且支持快速插入

- `map`中键值为`start`，`val`为`end`
- 利用二分查找找到集合中大于等于`start`最小数
- 若该数小于`end`，则说明集合重复了
  - 注意是小于，而不是小于等于，因为集合是左闭右开区间
- 迭代器前移一位，指向起始时间第一个小于等于`start`的日程
  - 若该日程终止时间大于`start`，则两集合重复
    - 注意是大于，而不是大于等于，因为集合是左闭右开区间

**代码**

```c++
class MyCalendar {
public:
    MyCalendar() {

    }
    
    bool book(int start, int end) {
        auto iter = mp.lower_bound(start);

        if(iter != mp.end() && iter->first < end)
            return false;

        if (iter != mp.begin() && ( -- iter)->second > start)
            return false;

        mp[start] = end;
        return true;
    }
private:
    map<int,int> mp;
};
```

## [剑指 Offer II 059. 数据流的第 K 大数值](https://leetcode-cn.com/problems/jBjn9C/)

**优先队列**

`priority_queue<int,vector<int>, greater<int>>`

```c++
class KthLargest {
public:
    priority_queue<int, vector<int>, greater<int>> q;
    int k;
    KthLargest(int k, vector<int>& nums) {
        this->k = k;
        for (auto& x: nums) {
            add(x);
        }
    }
    
    int add(int val) {
        q.push(val);
        if (q.size() > k) {
            q.pop();
        }
        return q.top();
    }
};
```

## [剑指 Offer II 060. 出现频率最高的 k 个数字](https://leetcode-cn.com/problems/g5c51o/)

**优先队列**

- 首先利用哈希表统计每个数字的频率
- 然后利用小根堆，根据频次维护小根堆

```c++
class Solution {
public:
    static bool cmp(pair<int, int>& m, pair<int, int>& n) {
        return m.second > n.second;
    }

    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> occurrences;
        for (auto& v : nums) {
            occurrences[v]++;
        }

        // pair 的第一个元素代表数组的值，第二个元素代表了该值出现的次数
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(&cmp)> q(cmp);
        for (auto& o : occurrences) {
            q.push(o);
            if(q.size() > k) q.pop();
        }
        vector<int> ret;
        while (!q.empty()) {
            ret.emplace_back(q.top().first);
            q.pop();
        }
        return ret;
    }
};
```

## [剑指 Offer II 061. 和最小的 k 个数对](https://leetcode-cn.com/problems/qn8gGX/)

对于一个最小数对`(nums1[i],nums2[j])`，下一个最小数对一定是`(nums1[i+1],nums2[j])`或者`(nums1[i],nums2[j+1])`

**但是如此添加会出现重复问题**

- 选择`(nums1[0],nums2[0])`以后，将`(nums1[0],nums2[1])`和`(nums1[1],nums2[0])`加入
- 选择`(nums1[0],nums2[1])`以后，将`(nums1[1],nums2[1])`和`(nums1[0],nums2[2])`加入
- 选择`(nums1[1],nums2[0])`以后，将`(nums1[1],nums2[1])`和`(nums1[2],nums2[1])`加入

以上步骤中出现了重复添加`(nums1[1],nums2[1])`的现象

为了解决该问题，我们可以先将`nums1`的前`k`个索引数对`(0,0),(1,0),(2,0)...(k-1,0)`加入队列，每次从队列取出对头元素`(x,y)`时，只需要将`nums2`的索引增加即可，这样避免了重复加入元素的问题

```c++
class Solution {
public:

    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        auto cmp = [&](const pair<int,int>& a, const pair<int,int>& b){
            return nums1[a.first] + nums2[a.second] > nums1[b.first] + nums2[b.second];
        };
        vector<vector<int>> res;
        priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> q(cmp);
        for(int i = 0; i < min((int)nums1.size(), k); i++){
            q.push(pair<int,int>{i,0});
        }
        while(k--){
            if(!q.empty()){
                auto top = q.top();
                q.pop();
                res.push_back(vector<int>{nums1[top.first],nums2[top.second]});
                if(top.second + 1 < nums2.size())
                    q.push(pair<int,int>{top.first, top.second + 1});
            }
        }
        return res;
    }
};
```



# 前缀树

## [剑指 Offer II 062. 实现前缀树](https://leetcode-cn.com/problems/QC3q1f/)

```c++
class Trie {
public:
    /** Initialize your data structure here. */
    Trie() {
        head = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode* cur = head;
        int i = 0;
        while(i < word.size()){
            int index = word[i] - 'a';
            if(cur->childred[index] != nullptr)
                cur = cur->childred[index];
            else{
                cur->childred[index] = new TrieNode();
                cur = cur->childred[index];
            }
            i++;
        }
        cur->isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        TrieNode* cur = head;
        int i = 0;
        while(i < word.size()){
            int index = word[i] - 'a';
            if(cur->childred[index] == nullptr)
                return false;
            cur = cur->childred[index];
            i++;
        }
        return cur->isEnd == true;
    }   
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        TrieNode* cur = head;
        int i = 0;
        while(i < prefix.size()){
            int index = prefix[i] - 'a';
            if(cur->childred[index] == nullptr)
                return false;
            cur = cur->childred[index];
            i++;
        }
        return true;
    }
private:
    struct TrieNode{
        TrieNode* childred[26];
        bool isEnd;
    };
    TrieNode* head;
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
```

**单独定义一个节点类型有点多余**

```c++
class Trie {
public:
    /** Initialize your data structure here. */
    Trie() {
        isEnd = false;
        memset(next,0,sizeof(next));
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        Trie* cur = this;
        for(char temp : word)
        {
            if(cur->next[temp-'a'] == NULL)
            {
                cur->next[temp-'a'] = new Trie();
            }
            cur = cur->next[temp-'a'];
        }
        cur->isEnd = true;
    }

    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        Trie* cur = this;
        for(char temp : word)
        {
            if(cur->next[temp-'a'] == NULL)
            {
                return false;
            }else{
                cur = cur->next[temp-'a'];
            }
        }
        return cur->isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        Trie* cur = this;
        for(char temp : prefix)
        {
            if(cur->next[temp-'a'] == NULL)
            {
                return false;
            }else{
                cur = cur->next[temp-'a'];
            }
        }
         return true;
    }

private:
    bool isEnd;
    Trie* next[26];
};
```



## [剑指 Offer II 063. 替换单词](https://leetcode-cn.com/problems/UhWRSj/)

**利用前缀树，首先将词根全部加入前缀树，然后依次搜寻给定sentence中的单词(该题需要自己拆分出单词)，若单词有前缀，直接返回前缀，否则返回单词本身**

**在上题实现的前缀树中添加一个方法用于寻找单词的前缀**

```c++
    string search(string word) {
        Trie* node = this;
        string prefix = "";
        for(char &c: word){
            if(node->isEnd) break;
            if(node->links[c - 'a'] == nullptr){
                return word;
            }
            prefix += c;
            node = node->links[c - 'a'];
        }
        return prefix;
    }
```

**代码：**

```c++
class Trie {
public:
    /** Initialize your data structure here. */
    Trie() {
        head = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode* cur = head;
        int i = 0;
        while(i < word.size()){
            int index = word[i] - 'a';
            if(cur->childred[index] != nullptr)
                cur = cur->childred[index];
            else{
                cur->childred[index] = new TrieNode();
                cur = cur->childred[index];
            }
            i++;
        }
        cur->isEnd = true;
    }
      
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        TrieNode* cur = head;
        int i = 0;
        while(i < prefix.size()){
            int index = prefix[i] - 'a';
            if(cur->childred[index] == nullptr)
                return false;
            cur = cur->childred[index];
            i++;
        }
        return true;
    }
    string searchPre(string word){
        TrieNode* cur = head;
        string res = "";
        int i = 0;
        while(i < word.size()){
            if(cur->isEnd) return res;
            int index = word[i] - 'a';
            if(cur->childred[index] == nullptr)
                return word;
            
            res += word[i++];
            cur = cur->childred[index];
        }
        return res;        
    }
private:
    struct TrieNode{
        TrieNode* childred[26];
        bool isEnd;
    };
    TrieNode* head;
};

class Solution {
public:
    string replaceWords(vector<string>& dictionary, string sentence) {
        Trie* trie = new Trie();
        for(auto s : dictionary)
            trie->insert(s);

        string temp;
        string res;
        for(int i = 0; i < sentence.size(); i++){
            if(sentence[i] != ' '){
                temp += sentence[i];
            }else{
                if(temp != ""){
                    res += trie->searchPre(temp);
                    if(i < sentence.size())
                        res += " ";                    
                }
                temp = "";
            }
        }
        //循环终止后，最后一个单词还未被处理
        if(temp != ""){
            res += trie->searchPre(temp);
        }
        return res;
    }
};
```

## [剑指 Offer II 064. 神奇的字典](https://leetcode-cn.com/problems/US1pGT/)

具体思路就是根据 dfs 搜索前缀树的每条路径。如果到达的节点与字符串中的字符不匹配，则表示此时需要修改该字符以匹配路径。如果到达对应字符串的最后一个字符所对应的节点，且该节点的 isWord 为 ture，并且当前路径刚好修改了字符串中的一个字符，那么就找到了符合要求的路径，返回 true

```c++
// 构造前缀树节点
class Trie {
public:
    bool isWord;
    vector<Trie*> children;
    Trie () : isWord(false), children(26, nullptr) {}

    void insert(const string& str) {
        Trie* node = this;
        for (auto& ch : str) {
            if (node->children[ch - 'a'] == nullptr) {
                node->children[ch - 'a'] = new Trie();
            }
            node = node->children[ch - 'a'];
        }
        node->isWord = true;
    }
};
class MagicDictionary {
private:
    Trie* root;
    bool dfs(Trie* root, string& word, int i, int edit) {
        if (root == nullptr) {
            return false;
        }

        if (root->isWord && i == word.size() && edit == 1) {
            return true;
        }

        if (i < word.size() && edit <= 1) {
            bool found = false;
            for (int j = 0; j < 26 && !found; ++j) {
                int next = (j == word[i] - 'a') ? edit : edit + 1;
                found = dfs(root->children[j], word, i + 1, next);
            }

            return found;
        }

        return false;
    }

public:
    /** Initialize your data structure here. */
    MagicDictionary() {
        root = new Trie();
    }
    
    void buildDict(vector<string> dictionary) {
        for (auto& word : dictionary) {
            root->insert(word);
        }
    }
    
    bool search(string searchWord) {
        return dfs(root, searchWord, 0, 0);
    }
};
```

## [剑指 Offer II 065. 最短的单词编码](https://leetcode-cn.com/problems/iSwD2y/)

**题意即：若有一个单词为另一单词的后缀，则忽略该单词**

解法一：利用一个哈希表存储所有单词，然后遍历所有单词，对于一个单词，遍历其所有后缀，若哈希表中存在一个该单词的后缀单词，则从哈希表中移除，最终哈希表中存储的元素都是**不是其他单词后缀**的单词，最后的结果就是这些单词长度加一的总和

```c++
class Solution {
public:
    int minimumLengthEncoding(vector<string>& words) {
        int res = 0;
        unordered_set<string> mp(words.begin(), words.end());
        for(auto word : words){
            for(int i = 1; i < word.size(); i++){
                if(mp.count(word.substr(i)))
                    mp.erase(word.substr(i));
            }
        }
        for(auto m : mp){
            res += m.size() + 1;
        }   
        return res;
    }
};
```

解法二：将单词反转后，插入前缀树中，此时后缀不就变成前缀了嘛，判断**一个单词为另一单词的后缀**就变成了与上一题一样的判断是否是一个单词的前缀

注意：

- 要先将给定数组按字符串按长度降序排序，先判断长字符串，再判断短的

```c++
class Trie {
public:
    /** Initialize your data structure here. */
    Trie() {
        head = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode* cur = head;
        int i = 0;
        while(i < word.size()){
            int index = word[i] - 'a';
            if(cur->childred[index] != nullptr)
                cur = cur->childred[index];
            else{
                cur->childred[index] = new TrieNode();
                cur = cur->childred[index];
            }
            i++;
        }
        cur->isEnd = true;
    }
      
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        TrieNode* cur = head;
        int i = 0;
        while(i < prefix.size()){
            int index = prefix[i] - 'a';
            if(cur->childred[index] == nullptr)
                return false;
            cur = cur->childred[index];
            i++;
        }
        return true;
    }
    bool searchPre(string word){
        TrieNode* cur = head;
        string res = "";
        int i = 0;
        while(i < word.size()){
            if(cur->isEnd) break;
            int index = word[i] - 'a';
            if(cur->childred[index] == nullptr)
                return false;
            
            res += word[i++];
            cur = cur->childred[index];
        }
        return true;        
    }
private:
    struct TrieNode{
        TrieNode* childred[26];
        bool isEnd;
    };
    TrieNode* head;
};


class Solution {
public:
    Trie* root = new Trie();    
    int minimumLengthEncoding(vector<string>& words) {
        sort(words.begin(),words.end(), [&](const string& a, const string& b){
            return a.size() > b.size();
        });
        string res = "";
        reverse(words[0].begin(),words[0].end());
        root->insert(words[0]);
        res += words[0];
        res += '#';
        for(int i = 1; i < words.size(); i++){
            reverse(words[i].begin(),words[i].end());
            if(root->searchPre(words[i])){
                continue;
            }else{
                res += words[i];
                res += '#';
                root->insert(words[i]);
            }
        }
        return res.size();
    }
};
```

## [剑指 Offer II 066. 单词之和](https://leetcode-cn.com/problems/z1R5dt/)



```c++
class Trie
{
    struct TrieNode{
        int val;
        TrieNode* children[26];
        TrieNode(int x)
        {
            val = x;
            for(int i = 0; i < 26; i++)
            {
                children[i] = nullptr;
            }
        }
    };
    TrieNode* root;
public:
    Trie()
    {
        root = new TrieNode(-1);
    }

    void insert(string& key,int val)
    {
        auto p = root;
        for(auto s : key)
        {
            if(!p->children[s - 'a'])
            {
                p->children[s - 'a'] = new TrieNode(0);
            }
            p = p->children[s - 'a'];
        }
        //尾字符记录val
        p->val = val;
    }

    int sum(string& prefix)
    {
        auto p = root;
        for(auto s : prefix)
        {
            if(!p->children[s - 'a'])   return 0;
            p = p->children[s - 'a'];
        }
        int sum = 0;
        queue<TrieNode*> q;
        q.push(p);
        while(!q.empty())
        {
            TrieNode* cur = q.front();
            sum += cur->val;
            q.pop();
            for(auto child : cur->children)
            {
                if(child)   q.push(child);
            }
        }
        return sum;
    }

};

class MapSum {
public:
    MapSum() {

    }
    
    void insert(string key, int val) {
        mapSum.insert(key,val);
    }
    
    int sum(string prefix) {
        return mapSum.sum(prefix);
    }
private:
    Trie mapSum;
};
```

## [剑指 Offer II 067. 最大的异或](https://leetcode-cn.com/problems/ms70jA)

**异或运算:**

- `0 ^ 1 = 1`
- `1 ^ 1 = 0`
- `0 ^ 0 = 0`

对于一个数`x`，要寻找与其异或后结果最大的`y`,即`max(x^y)`,

- 首先，二进制高位为`1`会大于低位的所有和
  - 即`2^n > 2^(n-1) + ... + 2^0`
  - 因此进行异或时尽量选择高位异或结果为`1`的数**(贪心)**
- 从高位遍历`x`
  - 若当前位为`1`,则`y`的当前位应该为`0`
  - 若当前位为`0`,则`y`的当前位应该为`1`

**利用前缀树，由于每一位非`0`即`1`,因此前缀树实际上是一棵二叉树**

```c++
class Trie{
	vector<Trie*> next[2];
};
```

- 首先将所有数字插入前缀树，从高位开始插入
- 然后遍历所有数字，依次寻找与其异或后结果最大的`y`,并返回结果`x^y`

```c++
class Trie{
    Trie* next[2]={nullptr};
public:
    Trie(){};

    void insert(int n){
        Trie* root = this;
        for(int i = 30; i >= 0; i--){
            int cur = n >> i & 1;
            if(!root->next[cur]){
                root->next[cur] = new Trie();
            }
            root = root->next[cur];
        }
    }

    int search(int x){
        Trie* root = this;
        int res = 0;
        for(int i = 30; i >= 0; i--){
            int n = (x >> i) & 1;
            if(root->next[!n]){
                root = root->next[!n];
                res = res * 2 + !n;
            }else{
                root = root->next[n];
                res = res * 2 + n;
            }
        }
        res ^= x;
        return res;
    }
};
class Solution {
public:
    Trie* root = new Trie();
    int findMaximumXOR(vector<int>& nums) {
        for(auto num : nums)
            root->insert(num);

        int res = 0;
        for(auto num : nums){
            res = max(res, root->search(num));
        }
        return res;

    }
};
```

# 二分法

*贴上一个讲解很详细的链接**

[我作了首诗，保你闭着眼睛也能写对二分查找 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485044&idx=1&sn=e6b95782141c17abe206bfe2323a4226&scene=21)

## [剑指 Offer II 068. 查找插入位置](https://leetcode-cn.com/problems/N6YdxV/)

### 左闭右闭区间写法(l <= r)

```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int n = nums.size();
        int l = 0, r = n - 1;
        while(l <= r){
            int mid = l + r >> 1;
            if(nums[mid] < target) l = mid + 1;
            else if(nums[mid] == target) return mid;
            else
                //右区间更新方式不同
                r = mid - 1;
        }
        return l;
    }
};
```

### 左闭右开区间写法(l < r)

```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int n = nums.size();
        int l = 0, r = n;
        while(l < r){
            int mid = l + r >> 1;
            if(nums[mid] < target) l = mid + 1;
            else if(nums[mid] == target) return mid;
            else
               	//右区间更新方式不同
                r = mid;
        }
        return l;
    }
};
```



## [剑指 Offer II 069. 山峰数组的顶部](https://leetcode-cn.com/problems/B1IidL/)

- 峰顶元素左侧满足 arr[i-1] < arr[i]性质，右侧不满足

- 峰顶元素右侧满足 arr[i] > arr[i+1]性质，左侧不满足

**第一种写法**

- 右边界初始化为`arr.size() - 1`
- 循环条件为`while(i <= j)`
- 更新边界
  - `i = mid + 1 || j = mid - 1`

```c++
class Solution {
public:
    int peakIndexInMountainArray(vector<int>& arr) {
        int i = 1, j = arr.size() - 1;
        int res = 0 ;
        while(i <= j){
            int mid = (i + j) >> 1;
            if(arr[mid] > arr[mid - 1]){
                res = mid;
                i = mid + 1;
            }
            else j = mid - 1;
        }
        return res;
    }
};
```

**另一种写法**

- 右边界初始化为`arr.size()`
- 循环条件为`while(i < j)`
- 更新边界
  - `i = mid + 1 || j = mid `

```c++
class Solution {
public:
    int peakIndexInMountainArray(vector<int>& arr) {
        int i = 1, j = arr.size();
        int res = 0 ;
        while(i < j){
            int mid = (i + j) >> 1;
            if(arr[mid] > arr[mid - 1]){
                res = mid;
                i = mid + 1;
            }
            else j = mid;
        }
        return res;
    }
};
```

## [剑指 Offer II 070. 排序数组中只出现一次的数字](https://leetcode-cn.com/problems/skFtm2)

**使用异或运算遍历数组，时间复杂度为`O(n)`，因此采用二分搜索将复杂度降至`o(logn)`**

给定数组中，只有一个元素`x`出现一次，其他元素均出现两次,则以`x`分界

- 对于`x`左边的元素，若有`nums[y] = nums[y + 1]`，则`y`必为偶数
- 对于`x`右边的元素，若有`nums[z] = nums[z + 1]`，则`z`必为奇数

**二分**

- 若`mid` 为偶数，则比较`nums[mid] == nums[mid + 1]`
  - 若相等，说明`mid`在`x`左边，要向右查找
  - 若不等，说明`mid`在`x`右边，要向左查找
- 若`mid` 为奇数，则比较`nums[mid] == nums[mid - 1]`
  - 若相等，说明`mid`在`x`左边，要向右查找
  - 若不等，说明`mid`在`x`右边，要向左查找

**细节**

利用按位异或的性质，可以得到 `mid `和相邻的数之间的如下关系，其中`⊕ `是按位异或运算符：

- 当`mid `是偶数时，`mid+1=mid⊕1`；


- 当`mid` 是奇数时，`mid−1=mid⊕1`。


```c++
//写法一：
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int n = nums.size();
        if(n == 1) return nums[0];
        int l = 0, r = n - 1;
        while(l <= r){
            int mid = (l + r) >> 1;
            if(nums[mid] == nums[mid ^ 1])
                l = mid + 1;
            else
                r = mid - 1;
        }
        return nums[l];
    }
};
//写法二
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int n = nums.size();
        if(n == 1) return nums[0];
        int l = 0, r = n;
        while(l < r){
            int mid = (l + r) >> 1;
            if(nums[mid] == nums[mid ^ 1])
                l = mid + 1;
            else
                r = mid;
        }
        return nums[l];
    }
};
```

## [剑指 Offer II 071. 按权重生成随机数](https://leetcode-cn.com/problems/cuyjEf/)

- 为题目给定的`nums`生成一个前缀和数组，以`[1, 2, 3, 5]`为例，其前缀和数组为`[1, 3, 6, 11]`，数组元素总和为`11`；


- 使用`rand() % 11`来生成一个`0~10`范围内的随机数，在前缀和数组中找到第一个严格大于它的数，那个数的下标就是我们要找的下标。因为权重越大的数，它所占据的区间范围越大（比如5所占据的区间为6~10），随机数也就越有可能落到它所在的区间内；


- 前缀和数组是有序的，有序数组中的查找可以使用二分法。




```c++
class Solution {
private:
    vector<int> sums;                       //sums为前缀和数组
    int total = 0;

public:
    Solution(vector<int>& w) {
        sums.resize(w.size());
        for (int i = 0; i < w.size(); ++ i) //构造前缀和数组
        {
            total += w[i];
            sums[i] = total;
        }
    }
    
    int pickIndex() {
        int rnd = rand() % total;           //生成最大值范围内的随机数
        int left = 0, right = sums.size();

        while (left < right)                //二分法在前缀和数组中找到第一个大于随机数的元素下标
        {
            int mid = left + (right - left) / 2;
            if (rnd < sums[mid])            
                right = mid;
            else
                left = mid + 1;
        }
        return left;
    }
};
/*
	另一种二分写法
    int pickIndex() {
        int rnd = rand() % total;           //生成最大值范围内的随机数
        int left = 0, right = sums.size() - 1;

        while (left <= right)                //二分法在前缀和数组中找到第一个大于随机数的元素下标
        {
            int mid = left + (right - left) / 2;
            if (rnd < sums[mid])            
                right = mid - 1;
            else
                left = mid + 1;
        }
        return left;
    }	
*/

//库函数版本
class Solution {
private:
    mt19937 gen;
    uniform_int_distribution<int> dis;
    vector<int> pre;

public:
    Solution(vector<int>& w): gen(random_device{}()), dis(1, accumulate(w.begin(), w.end(), 0)) {
        partial_sum(w.begin(), w.end(), back_inserter(pre));
    }
    
    int pickIndex() {
        int x = dis(gen);
        return lower_bound(pre.begin(), pre.end(), x) - pre.begin();
    }
};

```

## [剑指 Offer II 072. 求平方根](https://leetcode-cn.com/problems/jJ0w9p/)

```c++
class Solution {
public:
    int mySqrt(int x) {
        if(x <= 1) return x;
        int l = 0, r = x, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if ((long long)mid * mid <= x) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }
};
//另一种二分法
class Solution {
public:
    int mySqrt(int x) {
        if(x <= 1) return x;
        int l = 0, r = x, ans = -1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if ((long long)mid * mid <= x) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return ans;
    }
};

```

## [剑指 Offer II 073. 狒狒吃香蕉](https://leetcode-cn.com/problems/nZZqjQ/)

首先明确，速度一定是大于 1 的，并且小于香蕉的最大值，因此，可以采用二分查找，在`[1,max]`寻找最小速度

- 取中间速度`mid = 1 + max >> 1`
- 以速度`mid`吃完所有香蕉所需时间为 `t`
  - 若`t > h`：则速度太慢了，去`[mid+1,max]`中取
  - 若`t < h`：则速度太快了，去`[1,mid-1]`中取
  - 若`t == h`：去寻找可能存在的更小速度，因此去`[1,mid-1]`取

```c++
class Solution {
public:
    int getTime(vector<int>& piles, int v){
        int res = 0;
        for(int i = 0; i < piles.size(); i++){
            if(piles[i] / v == 0){
                res++;
            }else{
                res += piles[i] / v;
                if(piles[i] % v)
                    res++;
            }
        }
        return res;
    }

    int minEatingSpeed(vector<int>& piles, int h) {
        int mmax = 0;
        for(auto p : piles) mmax = max(mmax, p);
        int res = 0;
        int l = 1, r = mmax;
        while(l <= r){
            int mid = (r - l) / 2 + l;
            if(getTime(piles, mid) > h)
                l = mid + 1;
            else{
                res = mid;
                r = mid - 1;
            }                
        }
        return res;
    }
};
```

# 排序

## [剑指 Offer II 074. 合并区间](https://leetcode-cn.com/problems/SsGoHC/)

```c++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(),[&](const vector<int>& a, const vector<int>& b){
            if(a[0] == b[0]) return a[1] < b[1];
            return a[0] < b[0];
        });

        vector<vector<int>> res;
        res.push_back(intervals[0]);
        for(int i = 1; i < intervals.size(); i++){
            int n = res.size();
            if(res[n-1][1] >= intervals[i][0]){
                res[n-1][1] = max(res[n-1][1], intervals[i][1]);
            }else{
                res.push_back(intervals[i]);
            }
        }
        return res;
    }
};
```

## [剑指 Offer II 075. 数组相对排序](https://leetcode-cn.com/problems/0H97ZC/)

```c++
class Solution {
public:
    vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
        unordered_map<int, int> rank;
        for (int i = 0; i < arr2.size(); ++i) {
            rank[arr2[i]] = i;
        }
        sort(arr1.begin(), arr1.end(), [&](int x, int y) {
            if (rank.count(x)) {
                return rank.count(y) ? rank[x] < rank[y] : true;
            }
            else {
                return rank.count(y) ? false : x < y;
            }
        });
        return arr1;
    }
};
```

## [剑指 Offer II 076. 数组中的第 k 大的数字](https://leetcode-cn.com/problems/xx4gT2/)

**法一：快排**

**利用快速排序的性质：确定一个标志点，每轮从左右两边开始查找，首先从右往左遍历，找一个比标志点小的值right，然后从左往右遍历，找一个比标志点大的值left，然后交换两者，再然后把标志点与right交换，此时，标志点左侧的值都比标志点小，标志点右侧的值都比标志点大，然后以标志点为分界点，在左右子数组中继续以上过程**

**根据以上性质，每轮快排结束后，标志点左侧有i个元素，这i个元素均小于标志点，即标志点为第i个最小的元素，而要求第k个最大的元素，即求第 n - i个最小的元素，其中 n 为数组长度**

**快排时，一定要注意的一点是，每次快排时，一定要先从右向左查找，只有这样，在两个指针相遇时，指向的值一定比标志点小，在与标志点交换时才不会出错，因为每次以左边界点为标志点，交换时一定要保证交换值小于标志点**

```c++
class Solution {
public:
    int quickSort(vector<int>& nums, int l, int r, int k)
    {
        int i = l, j = r;
        while(i < j)
        {
            while(i < j && nums[j] >= nums[l]) j--;
            while(i < j && nums[i] <= nums[l]) i++;
            swap(nums[i],nums[j]);
        }
        swap(nums[l],nums[i]);

        if(i > k) return quickSort(nums,l,i-1,k);
        if(i < k) return quickSort(nums,i+1,r,k);

        return nums[i];
    }
    int findKthLargest(vector<int>& nums, int k) {
        return quickSort(nums,0,nums.size()-1,nums.size() - k);
    }
};
```

**法二：堆排序**

数组中的第K个最大元素即数组按降序排列后第 k 个元素，若利用堆排序来解决该问题，可参照如下思路：

- 小根堆——堆顶元素为最小元素，其他元素均小于该元素
- 大根堆——堆顶元素为最大元素，其他元素均大于该元素

直接利用大根堆的性质，先将数组所有元素存入大根堆，然后删除 k - 1 次堆顶元素，删除后的大根堆的堆顶元素就是答案。

**直接调库：**

```c++
class Solution 
{
public:
    int findKthLargest(vector<int>& nums, int k) 
    {
        priority_queue<int, vector<int>, less<int>> maxHeap;
        for (int x : nums)
            maxHeap.push(x);
        for (int i = 0; i < k - 1; i ++)
            maxHeap.pop();
        return maxHeap.top();
    }
};
```

**自己构建大根堆**

```c++
    const int N = 100010;
    int h[N];
    int idx;

    void down(int i, int n){
        int t = i;
        if(i * 2 <= n && h[2 * i] > h[t]) t = 2 * i;
        if(i * 2 + 1 <= n && h[2 * i + 1] > h[t]) t = 2 * i + 1;

        if(t != i){
            swap(h[t], h[i]);
            down(t,n);
        }
    }

    void up(int i, int n){
        while(i / 2 && h[i/2] < h[i]){
            swap(h[i/2], h[i]);
            i >>= 1;
        }
    }

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int n = nums.size();
        for(int i = 1; i <= n; i++){
            h[i] = nums[i-1];
        }
        for(int i = n/2; i > 0; i--)
            down(i,n);

        while(--k){
            h[1] = h[n--];
            down(1,n);
        }
        return h[1];
    }
};
```



## [剑指 Offer II 077. 链表排序](https://leetcode-cn.com/problems/7WHec2/)

**法一：自顶向下归并排序**

1. 找到链表中点（这里可以使用快慢指针），将链表分为两个子链表
2. 对两个子链表分别排序
3. 将两个排序后的子链表合并，得到完整的链表

可以用递归实现上述过程，递归终止条件为链表为空或者链表只包含一个节点

```c++
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        return sortList(head, nullptr);
    }

    ListNode* sortList(ListNode* head, ListNode* tail) {
        if (head == nullptr) {
            return head;
        }
        if (head->next == tail) {
            head->next = nullptr;
            return head;
        }
        ListNode* slow = head, *fast = head;
        while (fast != tail) {
            slow = slow->next;
            fast = fast->next;
            if (fast != tail) {
                fast = fast->next;
            }
        }
        ListNode* mid = slow;
        return merge(sortList(head, mid), sortList(mid, tail));
    }

    ListNode* merge(ListNode* head1, ListNode* head2) {
        ListNode* dummyHead = new ListNode(0);
        ListNode* temp = dummyHead, *temp1 = head1, *temp2 = head2;
        while (temp1 != nullptr && temp2 != nullptr) {
            if (temp1->val <= temp2->val) {
                temp->next = temp1;
                temp1 = temp1->next;
            } else {
                temp->next = temp2;
                temp2 = temp2->next;
            }
            temp = temp->next;
        }
        if (temp1 != nullptr) {
            temp->next = temp1;
        } else if (temp2 != nullptr) {
            temp->next = temp2;
        }
        return dummyHead->next;
    }
};
```

时间复杂度是`O(nlogn)`

空间复杂度是`O(logn)`，主要是递归产生的栈空间调用

## [剑指 Offer II 078. 合并排序链表](https://leetcode-cn.com/problems/vvXgSW/)

**分治**

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode *a, ListNode *b) {
        if ((!a) || (!b)) return a ? a : b;
        ListNode head, *tail = &head, *aPtr = a, *bPtr = b;
        while (aPtr && bPtr) {
            if (aPtr->val < bPtr->val) {
                tail->next = aPtr; aPtr = aPtr->next;
            } else {
                tail->next = bPtr; bPtr = bPtr->next;
            }
            tail = tail->next;
        }
        tail->next = (aPtr ? aPtr : bPtr);
        return head.next;
    }

    ListNode* merge(vector <ListNode*> &lists, int l, int r) {
        if (l == r) return lists[l];
        if (l > r) return nullptr;
        int mid = (l + r) >> 1;
        ListNode* a = merge(lists, l, mid);
        ListNode* b = merge(lists, mid + 1, r);
        return mergeTwoLists(a, b);
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return merge(lists, 0, lists.size() - 1);
    }
};
```

# 回溯

## [剑指 Offer II 079. 所有子集](https://leetcode-cn.com/problems/TVdhkn/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backTracking(vector<int>& nums, int startIndex){
        res.push_back(path);
        for(int i = startIndex; i < nums.size(); i++){
            path.push_back(nums[i]);
            backTracking(nums,i+1);
            path.pop_back();
        }
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        backTracking(nums,0);
        return res;
    }
};
```

## [剑指 Offer II 080. 含有 k 个元素的组合](https://leetcode-cn.com/problems/uUsW3B/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backTracking(int n, int k,int startIndex){
        if(path.size() == k){
            res.push_back(path);
            return;
        }
        for(int i = startIndex; i <= n; i++){
            path.push_back(i);
            backTracking(n,k,i + 1);
            path.pop_back();
        }
    }
    vector<vector<int>> combine(int n, int k) {
        backTracking(n,k,1);
        return res;
    }
};
```

## [剑指 Offer II 081. 允许重复选择元素的组合](https://leetcode-cn.com/problems/Ygoe9J/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backTracking(vector<int>& candidates, int target, int sum, int startIndex){
        if(sum == target){
            res.push_back(path);
            return;
        }
        if(sum > target) return;
        for(int i = startIndex; i < candidates.size(); i++){
            path.push_back(candidates[i]);
            sum += candidates[i];
            backTracking(candidates,target,sum,i);
            sum -= candidates[i];
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        backTracking(candidates,target,0,0);
        return res;
    }
};
```

## [剑指 Offer II 082. 含有重复元素集合的组合](https://leetcode-cn.com/problems/4sjJUc/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backTracking(vector<int>& candidates, int target, int sum, int startIndex, vector<bool>& used){
        if(sum == target){
            res.push_back(path);
            return;
        }
        if(sum > target) return;
        for(int i = startIndex; i < candidates.size(); i++){
            if(i > 0 && candidates[i] == candidates[i-1] && used[i-1] == false)
                continue;
            path.push_back(candidates[i]);
            sum += candidates[i];
            used[i] = true;
            backTracking(candidates,target,sum,i + 1,used);
            used[i] = false;
            sum -= candidates[i];
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        vector<bool> used(candidates.size(),false);
        backTracking(candidates,target,0,0,used);
        return res;
    }
};
```

## [剑指 Offer II 083. 没有重复元素集合的全排列](https://leetcode-cn.com/problems/VvJkup/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backTracking(vector<int>& nums,vector<bool>& used){
        if(path.size() == nums.size()){
            res.push_back(path);
            return;
        }

        for(int i = 0; i < nums.size(); i++){
            if(used[i]) continue;
            path.push_back(nums[i]);
            used[i] = true;
            backTracking(nums,used);
            used[i] = false;
            path.pop_back();
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<bool> used(nums.size(),false);
        backTracking(nums,used);
        return res;
    }
};
```

## [剑指 Offer II 084. 含有重复元素集合的全排列 ](https://leetcode-cn.com/problems/7p8L0Z/)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backTracking(vector<int>& nums,vector<bool>& used){
        if(path.size() == nums.size()){
            res.push_back(path);
            return;
        }

        for(int i = 0; i < nums.size(); i++){
            if(used[i]) continue;
            if(i > 0 && nums[i] == nums[i-1] && used[i-1] == false) continue;
            path.push_back(nums[i]);
            used[i] = true;
            backTracking(nums,used);
            used[i] = false;
            path.pop_back();
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        vector<bool> used(nums.size(),false);
        backTracking(nums,used);
        return res;
    }
};
```

## [剑指 Offer II 085. 生成匹配的括号](https://leetcode-cn.com/problems/IDBivT/)

```c++
class Solution {
public:
    vector<string> res;
    string path;
    void backTracking(int n, int left, int right){
        if(path.size() == 2 * n){
            res.push_back(path);
        }
        if(right > left) return;
        if(left < n){
            path += '(';
            backTracking(n,left + 1, right);
            path.pop_back();
        }
        if(right < left){
            path += ')';
            backTracking(n, left, right + 1);
            path.pop_back();
        }
    }
    vector<string> generateParenthesis(int n) {
        backTracking(n,0,0);
        return res;
    }
};
```

## [剑指 Offer II 086. 分割回文子字符串](https://leetcode-cn.com/problems/M99OJA/)

```c++
class Solution {
public:
    vector<vector<string>> res;
    vector<string> path;
    bool Ifpronme(const string& s, int i, int j)
    {
        if(i == j) return true;
        while(i < j)
        {
            if(s[i] != s[j]){
                return false;
            }else{
                i++;
                j--;
            }
        }
        return true;
    }
    void backTracking(string s, int startIndex)
    {
        if(startIndex >= s.size()){
            res.push_back(path);
            return;
        }
        for(int i = startIndex; i < s.size(); i++)
        {
            if(Ifpronme(s,startIndex,i)){
                string cur = s.substr(startIndex,i-startIndex+1);
                path.push_back(cur);
                backTracking(s,i+1);
                path.pop_back();
            }
        }
    }
    vector<vector<string>> partition(string s) {
        backTracking(s,0);
        return res;
    }
};
```

## [剑指 Offer II 087. 复原 IP ](https://leetcode-cn.com/problems/0on3uN/)

```c++
class Solution {
public:
    vector<string> res;
    bool isValid(const string& s, int start, int end)
    {
        if(start > end) return false;
        if(s[start] == '0' && start != end) return false;
        int num = 0;
        for(int i = start; i <= end; i++)
        {
            if(s[i] > '9' || s[i] <'0') return false;
            num = num * 10 + s[i] - '0';
            if(num > 255) return false;
        }
        return true;
    }
    void backtracking(string& s, int pointNum, int startIndex)
    {
        if(pointNum == 3){
            //判断最后一段字符是否满足条件
            if(isValid(s,startIndex,s.size() - 1)){
                res.push_back(s);
            }
            //如果最后一段不满足条件，直接返回
            //如果把return放在上面的if语句中，则会超时
            return;
        }
        for(int i = startIndex; i < s.size() ;i++)
        {
            if(isValid(s,startIndex,i)){
                s.insert(s.begin() + i + 1,'.');
                pointNum++;
                backtracking(s,pointNum,i + 2);
                pointNum--;
                s.erase(s.begin() + i + 1);
            }else{
                break;
            }
        }
    }

    vector<string> restoreIpAddresses(string s) {
        backtracking(s, 0, 0);
        return res;
    }
};
```

### 二刷：

```c++
class Solution {
public:
    int n;
    vector<string> res;
    bool isValid(const string& s, int start, int end)
    {
        if(start > end) return false;
        if(s[start] == '0' && start != end) return false;
        int num = 0;
        for(int i = start; i <= end; i++)
        {
            if(s[i] > '9' || s[i] <'0') return false;
            num = num * 10 + s[i] - '0';
            if(num > 255) return false;
        }
        return true;
    }
    // u代表当前搜到字符串的第几个位置
    // cnt为已经将字符串分割成了满足要求的几部分
    void dfs(string& s, int u, int cnt, string path)
    {
        if (u == n && cnt == 4)
        {
            path.pop_back();
            res.push_back(path);
            return;
        }
        if (u == n || cnt > 4)
            return;    // 剪枝
        for (int i = u; i < n; i ++ )
        {
            if(isValid(s,u,i)){
                string str = s.substr(u, i - u + 1);
                dfs(s, i + 1, cnt + 1, path + str + '.');
            }
            
        }
    }
    vector<string> restoreIpAddresses(string s) {
        n = s.size();
        dfs(s, 0, 0, "");
        return res;
    }

};
```



# 动态规划

## [剑指 Offer II 088. 爬楼梯的最少成本](https://leetcode-cn.com/problems/GzCJIP/)

`dp[i]`:爬到第 `i` 阶所需的最小体力

- 初始化：可以从下标为 0 或 1 的位置开始爬，即爬到这两个位置不需要花费体力
  - `dp[0] = 0;`
  - `dp[1] = 0;`
- 转移方程：每个台阶`i`可以由`i-1`或`i-2`一步爬到，每次爬的过程取体力花费最小的方案
  - `dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])`

```c++
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        vector<int> dp(n+1,0);
        dp[0] = 0;
        dp[1] = 0;
        for(int i = 2; i <= n; i++){
            dp[i] = min(dp[i-1] + cost[i-1],dp[i-2] + cost[i-2]);
        }
        return dp[n];
    }
};
```

## [剑指 Offer II 089. 房屋偷盗](https://leetcode-cn.com/problems/Gu0c2T/)

`dp[i]:`下标从0开始，从`0 ~ i `的房屋中可以偷窃的最大金额

- 初始化：
  - `dp[0] = nums[0]`
  - `dp[1] = max(nums[0],nums[1])`
- 状态转移：相邻房屋不可以偷，因此对于房屋`i`，有两种状态，一是偷该房屋，那么就不能偷`i-1`,而是不考虑该房屋，从`0~i-1`中考虑，即
  - `dp[i] = max(dp[i-2] + nums[i], dp[i-1])`

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if(n == 1) return nums[0];
        vector<int> dp(n,0);
        dp[0] = nums[0];
        dp[1] = max(nums[0],nums[1]);
        for(int i = 2; i < n; i++){
            dp[i] = max(dp[i-2] + nums[i], dp[i-1]);
        }
        return dp[n-1];
    }
};
```

## [剑指 Offer II 090. 环形房屋偷盗](https://leetcode-cn.com/problems/PzWKhm/)

与[剑指 Offer II 089. 房屋偷盗](https://leetcode-cn.com/problems/Gu0c2T/)类似，但是考虑到是环形，即多了一种情形，即首部房屋与尾部房屋不可以同时偷，因此

- 考虑偷头部房屋，则不能考虑尾部房屋，即在`0~nums.size()-2`中偷
- 考虑偷尾部房屋，则不能考虑头部房屋，即在`1~nums.size()-1`中偷

```c++
class Solution {
public:
    int rob(vector<int>& nums, int i , int j) {
        int n = nums.size();
        if(i == j) return nums[i];
        vector<int> dp(n,0);
        dp[i] = nums[i];
        dp[i+1] = max(nums[i],nums[i+1]);
        for(int k = i+2; k <= j; k++){
            dp[k] = max(dp[k-2] + nums[k], dp[k-1]);
        }
        return dp[j];
    }
    int rob(vector<int>& nums) {
        if(nums.size() == 1) return nums[0];
        return max(rob(nums,0,nums.size() - 2), rob(nums,1,nums.size() - 1));
    }
};
```

## [剑指 Offer II 091. 粉刷房子](https://leetcode-cn.com/problems/JEj789)

每个房子可以被粉刷成三种颜色，并且相邻房子的颜色不可以相同，即每个房子有三种状态，用二维数组分别表示这三种状态

- `dp[i][0]`：下标为`i`的房子粉刷为红色时，`0~i`栋房子所需的最小花销
  - 当前房子为红色时，前一栋房子不可以为红色
  - `dp[i][0] = min(dp[i-1][1], dp[i-1][2]) + costs[i][0]`
- `dp[i][1]`：下标为`i`的房子粉刷为绿色时，`0~i`栋房子所需的最小花销
  - `dp[i][1] = min(dp[i-1][0], dp[i-1][2]) + costs[i][1]`
- `dp[i][2]`：下标为`i`的房子粉刷为蓝色时，`0~i`栋房子所需的最小花销
  - `dp[i][2] = min(dp[i-1][0], dp[i-1][1]) + costs[i][2]`

```c++
class Solution {
public:
    int minCost(vector<vector<int>>& costs) {
        //房子数量
        int n = costs.size();
        if(n == 1) return min(costs[0][0], min(costs[0][1],costs[0][2]));
        //相邻房屋颜色不同，每一个房子有三种状态
        vector<vector<int>> dp(n,vector<int>(3,0));
        dp[0][0] = costs[0][0];
        dp[0][1] = costs[0][1];
        dp[0][2] = costs[0][2];

        for(int i = 1; i < n; i++){
            dp[i][0] = min(dp[i-1][1], dp[i-1][2]) + costs[i][0];
            dp[i][1] = min(dp[i-1][0], dp[i-1][2]) + costs[i][1];
            dp[i][2] = min(dp[i-1][0], dp[i-1][1]) + costs[i][2];
        }

        return min(dp[n-1][0], min(dp[n-1][1], dp[n-1][2]));
    }
};
```



## [剑指 Offer II 092. 翻转字符](https://leetcode-cn.com/problems/cyJERH/)

每个字符有两种状态，即 0 或 1，我们可以考虑将当前字符分别变为0 或 1所需的花销，然后取最小值，因此，定义动规数组如下：

- `dp[i][0]`：将`s[i]`变为 0 所需的最小花销
  - 若要满足递增条件，则前面的字符均要变为 0
  - `dp[i][0] = dp[i-1][0] + (s[i] == '1' ? 1 : 0)`
- `dp[i][1]`：将`s[i]`变为 1 所需的最小花销
  - 若要满足递增条件，则前面的字符可以全为 0 或者全为1，因此取两者中的最小值
  - `dp[i][1] = min(dp[i-1][0], dp[i-1][1]) + (s[i] == '1' ? 0 : 1)`

```c++
class Solution {
public:
    int minFlipsMonoIncr(string s) {
        int n = s.size();
        //dp[i][0]表示前i个元素，最后一个元素为0的最小翻转次数；
        //dp[i][1]表示前i个元素，最后一个元素为1的最小翻转次数
        vector<vector<int>> dp(n,vector<int>(2,0));
        dp[0][0] = s[0] == '0' ? 0 : 1;
        dp[0][1] = s[0] == '1' ? 0 : 1;

        for(int i = 1; i < n; i++){
            dp[i][0] = dp[i-1][0] + (s[i] == '1' ? 1 : 0);
            dp[i][1] = min(dp[i-1][0], dp[i-1][1]) + (s[i] == '1' ? 0 : 1);
        }

        return min(dp[n-1][0], dp[n-1][1]);
    }
};
```

**可以利用滚动数组优化**

```c++
class Solution {
public:
    int minFlipsMonoIncr(string s) {
        int n = s.size();
        //dp[i][0]表示前i个元素，最后一个元素为0的最小翻转次数；
        //dp[i][1]表示前i个元素，最后一个元素为1的最小翻转次数
        vector<int> dp(2,0);
        dp[0] = s[0] == '0' ? 0 : 1;
        dp[1] = s[0] == '1' ? 0 : 1;

        for(int i = 1; i < n; i++){
            //一定要先计算dp[1]，因为dp[1]会用到上一层的dp[0]
            //若先计算dp[0]，计算dp[1]时用的就是当前层的dp[0]
            dp[1] = min(dp[0], dp[1]) + (s[i] == '1' ? 0 : 1);
            dp[0] = dp[0] + (s[i] == '1' ? 1 : 0);
        }

        return min(dp[0], dp[1]);
    }
};
```

## [剑指 Offer II 093. 最长斐波那契数列](https://leetcode-cn.com/problems/Q91FMA/)

`dp[i][j]`:在给定数组`A[]`中，以`A[i],A[j]`为结尾两元素的斐波拉契数列的最大长度

- 即在`A[0]...A[i]..A[j]..`中，斐波拉契数列形式为`....A[i],A[j]`

**状态转移方程：**

- 考虑`A[i]`之前的数字中是否存在一个数`A[k]`，使得`A[k] + A[i] = A[j]`
- 即`dp[i][j] = dp[k][i] + 1 ，其中A[k]满足 A[k] + A[i] = A[j]`

**初始化：**

- 任意两个元素都是有效的斐波拉契数列，因此初始化`dp[i][j] = 2`，其中`i~(0,n), j~(i,n)`

**代码编写：**

- 由于给定数组是一个严格递增的序列，因此我们不需要从头到尾去遍历寻找`A[k]`,我们可以利用一个哈希表来快速寻找

```c++
class Solution {
public:
    int lenLongestFibSubseq(vector<int>& arr) {
        int res = 0;
        int n = arr.size();
        unordered_map<int,int> hash;
        //建立映射
        for(int i = 0; i < arr.size(); i++) hash[arr[i]] = i;
        vector<vector<int>> dp(n,vector<int>(n,0));
        //初始化
        for(int i = 0; i < n; i++){
            for(int j = i+1; j < n; j++)
                dp[i][j] = 2;
        }
        dp[0][0] = 1;
        for(int i = 1; i < n; i++){
            for(int j = i + 1; j < n; j++){
                if(hash.find(arr[j] - arr[i]) != hash.end()){
                    int k = hash[arr[j] - arr[i]];
                    if(k < i)
                    	dp[i][j] = max(dp[i][j], dp[k][i] + 1);
                    res = max(res, dp[i][j]);
                }
            }
        }
        //题目要求斐波拉契数列长度必须大于等于3
        return res > 2 ? res : 0;
    }
};
```

## [剑指 Offer II 094. 最少回文分割](https://leetcode.cn/problems/omKAoA/)

`dp[i]:`给定字符串中前`0~i`个字符组成的子字符串分割成回文子串的最小分割数

- 对于`dp[i]`,寻找一个最小的`dp[j]`，其中`0 <= j < i`,则`dp[i] = dp[j] + 1`
  - `s[j+1,...,i]`为回文串
- `s[0~i]`可能本身就是一个回文串，此时`dp[i] = 0`
- 借助[647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)，记录下任意子字符串是否为回文串

```c++
class Solution {
public:
    int minCut(string s) {
        int n = s.size();
        vector<vector<bool>> dp(n,vector<bool>(n,false));
        for(int i = n-1; i >= 0; i--){
            for(int j = i; j < n; j++){
                if(s[i] == s[j]){
                    if( j - i <= 1) dp[i][j] = true;
                    else{
                        dp[i][j] = dp[i+1][j-1];
                    }
                }
            }
        }

        vector<int> f(n,INT_MAX);
        for(int i = 0; i < n; i++){
            //如果本身是回文串
            if(dp[0][i]){
                f[i] = 0;
            }else{
                //如果本身不是回文串，则去寻找最小f[j]
                for(int j = 0; j < i; j++)
                    if(dp[j + 1][i])
                        f[i] = min(f[i], f[j] + 1);
            }
        }
        return f[n-1];
    }
};
```

## [剑指 Offer II 095. 最长公共子序列](https://leetcode.cn/problems/qJnOS7/)

`dp[i][j]`:`text1`前`i`个字符中与` text2`前`j`个字符中的最长公共子序列

```c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size();
        int n = text2.size();
        //dp[i][j]: text1前i个字符中与 text2前j个字符中的最长公共子序列
        vector<vector<int>> dp(m+1,vector<int>(n+1,0));
        for(int i = 1; i <= m; i++){
            for(int j = 1; j <= n; j++){
                if(text1[i-1] == text2[j-1])
                    dp[i][j] = dp[i-1][j-1] + 1;
                else{
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[m][n];
    }
};
```

## [剑指 Offer II 096. 字符串交织](https://leetcode.cn/problems/IY6buf/)*

**`dp[i][j]`:**

- 字符串`s1`中前`i`个字符组成的子字符串(下标为`0~i-1`的子字符串)**(长度为`i`)**与字符串`s2`中前`j`个字符组成的子字符串(下标为`0~j-1`的子字符串)**(长度为`j`)**能否交织成目标字符串`s`中前`i+j`个字符组成的子字符串(下标为`0~i+j-1`)**(长度为`i + j `)**

**状态转移：**

- `s[i+j-1]`可能来自于`s1[i-1]`，也可能来自于`s2[j-1]`
- 若`s[i+j-1] == s1[i-1]`，即`s[i+j-1]`来自于`s1[i-1]`
  - 则对于目标字符串`s`中`0~i+j-2`的子字符串，其是否能由字符串`s1`中**下标为`0~i-2`**的子字符串**(长度为`i-1`)**与字符串`s2`中**下标为`0~j-1`**的子字符串**(长度为`j`)**交织而成，决定`dp[i][j]`是否为`true`
  - `dp[i][j] = dp[i-1][j]`
- 若`s[i+j-1] == s2[j-1]`，即`s[i+j-1]`来自于`s2[j-1]`
  - 同上
  - `dp[i][j] = dp[i][j-1]`
- 综上：
  - `dp[i][j] = (s[i+j-1] == s1[i-1] && dp[i-1][j]) || (s[i+j-1] == s2[j-1] && dp[i][j-1])`

**初始化：**

- `dp[0][0] = true`
  - 即两个空字符串能否交织成空字符串，显然为真
- `dp[i][0] = s[i-1] == s1[i-1] && dp[i-1][0]`
  - 即当 `s2`为空时，只有当`s[i-1] == s1[i-1]`并且`dp[i-1][0] == true`时，`dp[i][0] = true`
- `dp[0][i] = s[i-1] == s2[j-1] && dp[0][i-1]`
  - 同上

**代码：**

```c++
class Solution {
public:
    bool isInterleave(string s1, string s2, string s) {
        if(s1.size() + s2.size() != s.size()) return false;
        int n1 = s1.size(), n2 = s2.size();
        vector<vector<int>> dp(n1+1, vector<int>(n2+1,false));
        dp[0][0] = true;
        //s[i-1] != s1[i-1]时直接跳出循环，后面的均为false
        for(int i = 1; i <= n1 && s[i-1] == s1[i-1]; i++) dp[i][0] = true;
        for(int i = 1; i <= n2 && s[i-1] == s2[i-1]; i++) dp[0][i] = true;

        for(int i = 1; i <= n1; i++){
            for(int j = 1; j <= n2; j++){
                dp[i][j] = (s[i+j-1] == s1[i-1] && dp[i-1][j]) || (s[i+j-1] == s2[j-1] && dp[i][j-1]);
            }
        }

        return dp[n1][n2];
    }
};
```



## [剑指 Offer II 097. 子序列的数目](https://leetcode.cn/problems/21dk04/)

- ```c++
  dp[i][j]:以i-1为结尾的s子序列中出现以j-1为结尾的t的个数为dp[i][j]。
  ```

- 这一类问题，基本是要分析两种情况

  - s[i - 1] 与 t[j - 1]相等
  - s[i - 1] 与 t[j - 1] 不相等

```c++
当s[i - 1] 与 t[j - 1]相等时，dp[i][j]可以有两部分组成。

一部分是用s[i - 1]来匹配，那么个数为dp[i - 1][j - 1]。

一部分是不用s[i - 1]来匹配，个数为dp[i - 1][j]。
例如： s：bagg 和 t：bag ，s[3] 和 t[2]是相同的，但是字符串s也可以不用s[3]来匹配，即用s[0]s[1]s[2]组成的bag。

当然也可以用s[3]来匹配，即：s[0]s[1]s[3]组成的bag。
    
所以当s[i - 1] 与 t[j - 1]相等时，dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];

当s[i - 1] 与 t[j - 1]不相等时，dp[i][j]只有一部分组成，不用s[i - 1]来匹配，即：dp[i - 1][j]

所以递推公式为：dp[i][j] = dp[i - 1][j];
```

**该题还要注意整型溢出的问题，使用长整型**

```c++
  vector<vector<uint64_t>> dp(ns + 1,vector<uint64_t>(nt + 1,0));
```

```c++
class Solution {
public:
    int numDistinct(string s, string t) {
        int ns = s.size();
        int nt = t.size();
        vector<vector<uint64_t>> dp(ns + 1,vector<uint64_t>(nt + 1,0));
        for(int i = 0; i <= ns; i++) dp[i][0] = 1;
        for(int i = 1; i <= nt; i++) dp[0][i] = 0;

        for(int i = 1; i <= ns; i++){
            for(int j = 1; j <= nt; j++){
                if(s[i-1] == t[j-1])
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j];
                else
                    dp[i][j] = dp[i-1][j];
            }
        }
        return dp[ns][nt];
    }
};
```

**一维DP**

```c++
class Solution {
public:
    int numDistinct(string s, string t) {
        int ns = s.size();
        int nt = t.size();
        if(ns < nt) return 0;
        vector<uint64_t> dp(nt+1,0);
        dp[0] = 1;
        for(int i = 1; i <= ns; i++){
            for(int j = nt; j >= 1; j--){
                if(s[i-1] == t[j-1]){
                    dp[j] += dp[j-1];
                }
            }
        }
        return dp[nt];
    }
};
```



## [剑指 Offer II 098. 路径的数目](https://leetcode.cn/problems/2AoeFn/)

`dp[i][j]`:到达坐标`(i,j)`处有`dp[i][j]`种路径

**转移方程：**

- 考虑到只能向右或向下移动，对于每一个位置，只能从起上方或左方移动到
- 因此，到达指定点`(i,j)`的路径可以分为从上方到达以及从左方到达两种
- `dp[i][j] = dp[i-1][j] + dp[i][j-1]`

**初始化:**

- 横坐标或者纵坐标为`0`时，只能从其左边或者右边到达，全初始化为`1`

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m,vector<int>(n,0));
        for(int i = 0; i < n; i++) dp[0][i] = 1;
        for(int i = 1; i < m; i++) dp[i][0] = 1;

        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
```



## [剑指 Offer II 099. 最小路径之和](https://leetcode.cn/problems/0i0mDW/)

`dp[i][j]`:到达坐标`(i,j)`处所需花费的最小值

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();

        vector<vector<int>> dp(m,vector<int>(n,0));
        dp[0][0] = grid[0][0];
        for(int i = 1; i < n; i++) dp[0][i] = grid[0][i] + dp[0][i-1];
        for(int j = 1; j < m; j++) dp[j][0] = grid[j][0] + dp[j-1][0];

        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
        return dp[m-1][n-1];
    }
};
```

## [剑指 Offer II 100. 三角形中最小路径之和](https://leetcode.cn/problems/IlPe0q/)

**将三角形看作矩形的下半边**

|  2   |      |      |      |
| :--: | :--: | :--: | :--: |
|  3   |  4   |      |      |
|  6   |  5   |  7   |      |
|  4   |  1   |  8   |  3   |

`dp[i][j]`:到达坐标`(i,j)`的最小路径

**状态转移：**

- 每一步只能移动到下一行中相邻的节点上，即`(i,j)`只能移动到`(i+1,j)`或者`(i+1,j+1)`
- `dp[i][j] = min(dp[i-1][j], dp[i-1][j-1]) + triangle[i][j]`

**初始化：**

- 对于左边界上的点，即纵坐标全为`0`的点，只能由上方的点到达
  - `dp[i][0] = dp[i-1][0] + triangle[i-1][0]`
- 对于对角线上的点，即横纵坐标相等的点，只能由上一个的对角线点到达
  - `dp[i][i] = dp[i-1][i-1] + triangle[i][i]`

**结果：**

遍历到达最后一行节点的所有路径，返回最小值

```c++
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        //dp[i][j] = min(dp[i-1][j],dp[i-1][j-1]) + triangle[i][j]
        vector<vector<int>> dp(n,vector<int>(n,0));
        dp[0][0] = triangle[0][0];
        for(int i = 1; i < n; i++) dp[i][0] = dp[i-1][0] + triangle[i][0];
        for(int i = 1; i < n; i++) dp[i][i] = dp[i-1][i-1] + triangle[i][i];

        for(int i = 2; i < n; i++){
            for(int j = 1; j < i ; j++){
                dp[i][j] = min(dp[i-1][j], dp[i-1][j-1]) + triangle[i][j];
            }
        }

        int res = INT_MAX;
        for(auto d : dp[n-1]) res = min(res,d);
        return res;

    }
};
```

**可以利用滚动数组优化**

```c++
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        vector<int> dp(n,0);
        dp[0] = triangle[0][0];

        for(int i = 1; i < n; i++){
            for(int j = i; j >= 0; j--){
                //处理纵坐标为 0 的元素，只能从上一个点到达
                if(j == 0){
                    dp[j] = dp[j] + triangle[i][j];
                }
                //处理对角线元素，只能从上一个对角线元素到达
                else if(j == i)
                    dp[j] = dp[j-1] + triangle[i][j];
                else
                    dp[j] = min(dp[j], dp[j-1]) + triangle[i][j];
            }
        }

        int res = INT_MAX;
        for(auto d : dp) res = min(res,d);
        return res;
    }
};
```

## [剑指 Offer II 101. 分割等和子集](https://leetcode.cn/problems/NUPfPr/)

### 二维数组

```c++
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        int n = nums.size();
        for(auto x : nums)  sum += x;
        if(sum % 2 == 1) return false;
        int con = sum / 2;

        vector<vector<int>> dp(n+1,vector<int>(con+1,0));

        for(int i = nums[0]; i <= con; i++){
            dp[0][i] = nums[0];
        }
    
        for(int i = 1; i < n; i++){
            for(int j = 0; j <= con; j++){  
                if( j >= nums[i] )
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-nums[i]] + nums[i]);
                else
                    dp[i][j] = dp[i-1][j];  
            }
        }
        if(dp[n-1][con] == con) return true;
        return false;
    }
};

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for (auto& n : nums) {
            sum += n;
        }
        if ((sum & 1) != 0) {
            return false;
        }

        int target = sum >> 1;
        vector<vector<bool>> dp(nums.size() + 1, vector<bool>(target + 1, false));
        dp[0][0] = true;
        for (int i = 1; i <= nums.size(); ++i) {
            for (int j = 0; j <= target; ++j) {
                dp[i][j] = dp[i - 1][j];
                if (!dp[i][j] && j >= nums[i - 1]) {
                    dp[i][j] = dp[i - 1][j - nums[i - 1]];
                }
            }
        }
        return dp[nums.size()][target];
    }
};

```

### 一维数组

```c++
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        int n = nums.size();
        for(auto x : nums)  sum += x;
        if(sum % 2 == 1) return false;
        int con = sum / 2;

        vector<int> dp(con+1,0);

        for(int i = 0; i < n; i++){
            for(int j = con; j >= nums[i]; j--){  
                dp[j] = max(dp[j],dp[j-nums[i]] + nums[i]);
            }
        }
        if(dp[con] == con) return true;
        return false;
    }
};
```



## [剑指 Offer II 102. 加减的目标值](https://leetcode.cn/problems/YaVDxD/)

- 背包容量为 x = (target + sum) / 2
- 物品体积以及价值均为num[i]
- 物品数量为nums.size()
- dp[i]表示填满容量为i的背包有dp[i]种方法
- 递归公式：

```
因为填满容量为j-nums[i]的背包有dp[j-nums[i]]种方法
所以遍历到nums[i]时，填满容量为j的背包的方法就有dp[j-nums[i]]种方法，
即在填满容量为j-nums[i]的背包的每一种方法上在加上一个nums[i]

因此 dp[j] += dp[j-nums[i]];

```

**该公式在利用背包解决排列组合问题时经常会用到**

```c++
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        int sum = 0;
        for (int i = 0; i < nums.size(); i++) sum += nums[i];
        if (abs(S) > sum) return 0; // 此时没有方案
        if ((S + sum) % 2 == 1) return 0; // 此时没有方案
        int bagSize = (S + sum) / 2;
        vector<int> dp(bagSize + 1, 0);
        dp[0] = 1;
        for (int i = 0; i < nums.size(); i++) {
            for (int j = bagSize; j >= nums[i]; j--) {
                dp[j] += dp[j - nums[i]];
            }
        }
        return dp[bagSize];
    }
};
```

**二维数组**

```c++
class Solution {
public:
    /*
        x - y = target
        x + y = sum;
        x = (sum + target) / 2;
    */
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum = 0;
        int n = nums.size();
        for(auto num : nums) sum += num;
        if((sum + target) % 2) return 0;
        int k = sum + target >> 1;
        vector<vector<int>> dp(n+1, vector<int>(k+1,0));
        dp[0][0] = 1;
        for(int i = 1; i <= n; i++){
            for(int j = 0; j <= k; j++){
                dp[i][j] += dp[i-1][j];
                if(j >= nums[i-1])
                    dp[i][j] += dp[i-1][j - nums[i-1]];
            }
        }
        return dp[n][k];
    }
};
```

## [剑指 Offer II 103. 最少的硬币数目](https://leetcode.cn/problems/gaM7Ch/)

**二维数组**

`dp[i][j]`:从前 i 个 物品中填满容量为 j 的背包所需的最少物品个数

**状态转移：**

- 对于当前物品`coins[j]`,可以考虑将其加入背包与不加入背包两种情况
  - 加入背包`dp[i][j] = dp[i][j-coins[i-1]] + 1`
  - 不加入背包`dp[i][j] = dp[i-1][j]`
- 取两者中的最小值

**初始化**

- 因为求得是最少硬币数，所以初始化时应将数组内元素初始化为一个较大值

- 对于背包容量为0的情况，最少硬币数目为0

- ```c++
  vector<vector<int>> dp(n+1,vector<int>(amount+1,amount+1));
  for(int i = 0; i <= coins.size(); i++){
  	dp[i][0] = 0;
  }
  ```

```c++
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<vector<int>> dp(n+1,vector<int>(amount+1,amount+1));
        for(int i = 0; i <= coins.size(); i++){
            dp[i][0] = 0;
        }
        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= amount; j++){
                if(j >= coins[i - 1]){
                    dp[i][j] = min(dp[i - 1][j], 1 + dp[i][j - coins[i - 1]]);
                }else{
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n][amount] == amount+1 ? -1 : dp[n][amount];
    }
};
```

**一维**

**该题和纯背包问题不同，纯完全背包是能否凑成总金额，而本题是要求凑成总金额的个数**

并且题目描述中是凑成总金额的硬币**组合数**，**而非排列数**，排列和组合的区别在学习回溯算法的时候已经区别过了

```c++
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount+1,0);
        dp[0] = 1;
        for(int i = 0; i < coins.size(); i++){
            for(int j = coins[i]; j <= amount; j++){
                dp[j] += dp[j-coins[i]];
            }
        }
        return dp[amount];
    }
};
```

### 特别注意：

**该题对于遍历顺序有特殊要求，他只能遍历物品在外，遍历容积在内**

因为纯完全背包求得是否凑成总和，和凑成总和的元素有没有顺序没关系，即：有顺序也行，没有顺序也行！

而本题要求凑成总和的组合数，元素之间要求没有顺序。

所以纯完全背包是能凑成总和就行，不用管怎么凑的。

本题是求凑出来的方案个数，且每个方案个数是为组合数。

那么本题，两个for循环的先后顺序可就有说法了。

我们先来看 外层for循环遍历物品（钱币），内层for遍历背包（金钱总额）的情况。

代码如下：

```cpp
for (int i = 0; i < coins.size(); i++) { // 遍历物品
    for (int j = coins[i]; j <= amount; j++) { // 遍历背包容量
        dp[j] += dp[j - coins[i]];
    }
}
```

假设：coins[0] = 1，coins[1] = 5。

那么就是先把1加入计算，然后再把5加入计算，得到的方法数量只有{1, 5}这种情况。而不会出现{5, 1}的情况。

**所以这种遍历顺序中dp[j]里计算的是组合数！**

如果把两个for交换顺序，代码如下：

```c++
for (int j = 0; j <= amount; j++) { // 遍历背包容量
    for (int i = 0; i < coins.size(); i++) { // 遍历物品
        if (j - coins[i] >= 0) dp[j] += dp[j - coins[i]];
    }
}
```

背包容量的每一个值，都是经过 1 和 5 的计算，包含了{1, 5} 和 {5, 1}两种情况。

**此时dp[j]里算出来的就是排列数！**

**例如，1 和 3 都在数组nums 中，计算 dp[4] 的时候，排列的最后一个元素可以是 1 也可以是 3，因此 dp[1] 和 dp[3] 都会被考虑到，即不同的顺序都会被考虑到。**

## [剑指 Offer II 104. 排列的数目](https://leetcode.cn/problems/D0F0SV/)

```c++
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<int> dp(target + 1, 0);
        dp[0] = 1;

        for (int i = 1; i <= target; ++i) {
            for (auto& n : nums) {
                if (i >= n && dp[i - n] <= INT_MAX - dp[i]) {
                    dp[i] += dp[i - n];
                }
            }
        }
        return dp[target];
    }
};
```

# 图论DFS&BFS

## [剑指 Offer II 105. 岛屿的最大面积](https://leetcode.cn/problems/ZL6zAn/)

**遍历矩阵中每一个点，找到 1 时，深度优先遍历其上下左右四个方位，累加岛屿面积**

```c++
class Solution {
public:
    int dfs(vector<vector<int>>& grid, int i, int j, int m, int n){
        if(i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0) return 0;
        int res = 1;
        grid[i][j] = 0;

        int dx[] = {0,1,0,-1};
        int dy[] = {1,0,-1,0};
        for(int k = 0; k < 4; k++){
            int a = i + dx[k];
            int b = j + dy[k];
            if(a >= 0 && a < m && b >= 0 && b < n && grid[a][b] == 1){
                res += dfs(grid,a,b,m,n);
            }
        }
        return res;
    }
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int m = grid.size();
        if(!m) return 0;
        int n = grid[0].size();
        if(!n) return 0;
        
        int res = 0;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(grid[i][j] == 1)
                    res = max(res, dfs(grid,i,j,m,n));
            }
        }
        return res;
    }
};
```

## [剑指 Offer II 106. 二分图](https://leetcode.cn/problems/vEAB3K/)

### **染色法**

```c++
class Solution {
private:
    static constexpr int UNCOLORED = 0;
    static constexpr int RED = 1;
    static constexpr int GREEN = 2;
    vector<int> color;
    bool valid;

public:
    void dfs(int node, int c, const vector<vector<int>>& graph) {
        color[node] = c;
        int cNei = (c == RED ? GREEN : RED);
        for (int neighbor: graph[node]) {
            if (color[neighbor] == UNCOLORED) {
                dfs(neighbor, cNei, graph);
                if (!valid) {
                    return;
                }
            }
            else{
                if(color[neighbor] != cNei) {
                    valid = false;
                    return;
                }
            }
        }
    }

    bool isBipartite(vector<vector<int>>& graph) {
        int n = graph.size();
        valid = true;
        color.assign(n, UNCOLORED);
        for (int i = 0; i < n && valid; ++i) {
            if (color[i] == UNCOLORED) {
                dfs(i, RED, graph);
            }
        }
        return valid;
    }
};
```

### 并查集：

```c++
class Solution {
public:
    int p[101];
    int find(int x){
        if(p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    void merge(int i, int j){
        int p1 = find(i);
        int p2 = find(j); 
        if(p1 != p2){
            p[p1] = p2;
        }
    }
    bool isBipartite(vector<vector<int>>& graph) {
        int n = graph.size();
        for(int i = 0; i < n; i++){
            p[i] = i;
        }
        for(int i = 0; i < n; i++){
            for(int j = 0; j < graph[i].size(); j++){
                int p1 = find(i);
                int p2 = find(graph[i][j]);
                if(p1 == p2) return false;
                merge(graph[i][0], graph[i][j]);
            }
        }
        return true;
    }
};
```



## [剑指 Offer II 107. 矩阵中的距离](https://leetcode.cn/problems/2bCMpM/)

**最短路径问题，以每个0作为起点，搜寻到每个1的最短路径**

- 定义一个距离数组`dist[i][j]`:表示点`(i,j)`距离最近0的距离
  - 初始化: 若`mat[i][j] == 0`，则`dist[i][j] = 0`，否则，`dist[i][j] = INT_MAX`
- 以所有`mat[i][j] == 0`的点为起点，进行BFS，BFS通过队列实现
  - 将所有`mat[i][j] == 0`的点`(i,j)`加入队列
  - 取队列头，搜索其上下左右四个方向的点`(a,b)`
  - 若`dist[a][b] == INT_MAX`,表示其还未被搜索过，故更新其为`dist[i][j] + 1`
  - 确定`(a,b)`到0的最短距离以后，将`(a,b)`也作为起点加入队列
- BFS保证每一轮搜索都将距离0最近的点的最短距离进行了更新
  - 第一轮搜索距离为0的点(初始化)
  - 第二轮搜索时，以第一轮的点为起点，搜索上下左右相邻点，此时搜索到点的距离必为1
  - 第三轮搜索则是以第二轮搜索到的点为起点，此时搜索到的点距离必为2
  - 以此类推

```c++
class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int m = mat.size();
        int n = mat[0].size();
        queue<pair<int,int>> q;
        //距离数组
        vector<vector<int>> dists(m,vector<int>(n,INT_MAX));
        //初始化
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(mat[i][j] == 0){
                    dists[i][j] = 0;
                    q.push({i,j});
                }
            }
        }
        int dx[] = {0,-1,0,1};
        int dy[] = {1,0,-1,0};

        while(!q.empty()){
            int x = q.front().first;
            int y = q.front().second;
            q.pop();
            for(int k = 0; k < 4; k++){
                int a = x + dx[k];
                int b = y + dy[k];
                if(a < 0 || a >= m || b < 0 || b >= n) continue;
                //若该点未搜索过
                if(dists[a][b] == INT_MAX){
                    dists[a][b] = dists[x][y] + 1;
                    q.push({a,b});
                }
            }
        }
        return dists;
    }
};
```

## [剑指 Offer II 108. 单词演变](https://leetcode.cn/problems/om3reC/)

### 单向BFS

**建图：**

- 若两单词可以相互演变，则两个单词之间存在一条边
  - 演变：改变一个字母之后，两单词相同，即两单词对应位置上只有一个字母不同
- 题意即**求解给定两单词的之间的最短路径**
- 求解最短路径应用BFS

**算法流程：**

- 从`beginWord`开始搜寻
- 每次找出与`beginWord`距离最近的单词
  - 第一次找距离为1的单词
  - 第二次找距离为2的单词
- 直到找到`endWord`,此时层序遍历的层数即为路径

```c++
class Solution {
public:
    //寻找距离为1的单词，并将其加入队列
    void getNeighbor(string word, vector<string>& wordList, unordered_map<string,int>& mp, queue<string>& q){
        for(int i = 0; i < word.size(); i++){
            char temp = word[i];
            for(char ch = 'a'; ch <= 'z'; ch ++){
                word[i] = ch;
                if(ch != temp && mp.count(word)){
                    q.push(word);
                }
            }
            word[i] = temp;
        }
    }
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        //mp 记录当前节点是否被访问过
        unordered_map<string,int> mp;
        for(auto word : wordList) mp[word]++;
        queue<string> q;
        q.push(beginWord);
        int level = 0;
        while(!q.empty()){
            int size = q.size();
            level++;
            for(int i = 0; i < size; i++){
                string cur = q.front();
                q.pop();
                //j
                mp.erase(cur);
                if(cur == endWord){
                    return level;
                }
                getNeighbor(cur, wordList,mp,q);
            }
        }
        return 0;
    }
};
```

### 双向BFS

**从endWord与beginWord一起开始寻找**

- 定义3个集合s1,s2,s3
  - s1存储从beginWord开始BFS需要访问的节点
  - s2存储从endWord开始BFS需要访问的节点
  - s3存储与当前节点距离为1的节点，即当前节点的下一层需要访问的节点
- 每次从s1,s2中选择数量少的集合，进行搜索，这样可以缩小搜索空间
- 搜索时，若下一层单词在另一集合中出现了，说明两方向上有重合节点，该路径即为最短路径

```c++
class Solution {
private:
    bool getNeighbor(unordered_set<string>& visted, unordered_set<string>& st1, unordered_set<string>& st2, unordered_set<string>& st3, string& word) {
        for (int i = 0; i < word.size(); ++i) {
            char temp = word[i];
            for (char ch = 'a'; ch <= 'z'; ++ch) {
                word[i] = ch;
                if (ch != temp && visted.count(word)) {
                    //如果当前的邻点在st2中已经存在，说明已经存在一条begin到end的边
                    if (st2.count(word)) {
                        return true;
                    }
                    //st3
                    st3.insert(word);
                }
            }
            word[i] = temp;
        }
        return false;
    }

public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> visted;
        for (auto& word : wordList) {
            visted.insert(word);
        }
        if (!visted.count(endWord)) {
            return 0;
        }

        unordered_set<string> st1;
        unordered_set<string> st2;
        st1.insert(beginWord);
        st2.insert(endWord);
        int len = 2;

        while (!st1.empty() && !st2.empty()) {
            if (st1.size() > st2.size()) {
                swap(st1, st2);
            }
            //st3存储当前点的相邻点
            unordered_set<string> st3;
            for (auto it = st1.begin(); it != st1.end(); ++it) {
                string word = *it;
                visted.erase(word);

                if (getNeighbor(visted, st1, st2, st3, word)) {
                    return len;
                }
            }
            st1 = st3;
            len++;
        }

        return 0;
    }
};
```

### 堆优化Dijkstra算法

```c++
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        int n=wordList.size();
        int wordLen=beginWord.size();
        //当前单词到beginword的最短路径
        unordered_map<string,int> wordPathLen;
        //初始化距离向量，除了beginWord之外，其他单词距离初始化为INT_MAX/2
        for(int i=0;i<n;i++){
            wordPathLen[wordList[i]]=INT_MAX/2;
        }
        //PS：这里一定要后初始化beginWord，因为wordList里面可能有beginWord
        wordPathLen[beginWord]=1;
        //wordQ记录当前未访问的结点，要求路径最短的结点在顶部，所以用小顶堆存储
        auto cmp=[&](const string& a, const string& b){
            return wordPathLen[a]>wordPathLen[b];
        };
        priority_queue<string,vector<string>,decltype(cmp)> wordQ(cmp);
        wordQ.push(beginWord);
        //wordList中不存在endword，直接返回
        if(wordPathLen.count(endWord)==0) return 0;        
        while(!wordQ.empty()){
            string word=wordQ.top();
            wordQ.pop();
            string originWord=word;
            //寻找与当前单词距离为1的单词
            for(int i=0;i<wordLen;i++){
                //记录原始字符
                char originAlpha=word[i];
                for(char j='a';j<='z';j++){
                    if(j==originAlpha) continue;
                    word[i]=j;
                    //判断单词是否在wordList中
                    if(wordPathLen.count(word)==0) continue;
                    //判断是否为endWord
                    if(word==endWord) {
                        return wordPathLen[originWord]+1;
                    }
                    //更新最短距离
                    if(wordPathLen[originWord]+1 < wordPathLen[word]){
                        if(wordPathLen[word] == INT_MAX/2) {
                            wordQ.push(word);
                        }
                        wordPathLen[word]=wordPathLen[originWord] + 1;
                    }
                }
                //一定要进行初始化，因为要让word回到原始状态，才能修改其他位的字母
                word=originWord;
            }
        }
        return 0;
    }
};
```

## [剑指 Offer II 109. 开密码锁](https://leetcode.cn/problems/zlDJc7/)

### 单向BFS

```c++
class Solution {
public:
    unordered_set<string> visit;
    void getNeighbor(string& word, queue<string>& q, string& target){
        for(int i = 0; i < word.size(); i++){
            char temp = word[i];
            string v(2,' ');
            v[0] = temp == '9' ? '0' : temp + 1;
            v[1] = temp == '0' ? '9' : temp - 1;
            for(int j = 0; j < 2; j++){
                word[i] = v[j];
                if(visit.count(word)) continue;
                visit.insert(word);
                q.push(word);
            }
            word[i] = temp;
        }
    }
    int openLock(vector<string>& deadends, string target) {
        for(auto s : deadends) visit.insert(s);
        if(visit.count("0000")) return -1;
        queue<string> q;
        q.push("0000");
        visit.insert("0000");
        int res = 0;
        while(!q.empty()){
            int size = q.size();
            for(int i = 0; i < size; i++){
                string cur = q.front();
                q.pop();
                if(cur == target) return res;
                getNeighbor(cur, q, target);
            }
            res++;
        }
        return -1;
    }
};
```

### 双向BFS

```c++
class Solution {
private:
    bool helper(string& word, unordered_set<string>& visted, unordered_set<string>& st2, unordered_set<string>& st3) {
        for (int i = 0; i < word.size(); ++i) {
            char temp = word[i];
            string var(2,' ');
            var[0] = (temp + 1 > '9') ? '0': temp + 1;
            var[1] = (temp - 1 < '0') ? '9': temp - 1;
            for (auto& ch : var) {
                word[i] = ch;
                if (!visted.count(word)) {
                    if (st2.count(word)) {
                        return true;
                    }
                    st3.insert(word);
                }
            }
            word[i] = temp;
        }
        return false;
    }

public:
    int openLock(vector<string>& deadends, string target) {
        string init = "0000";
        if (target == init) {
            return 0;
        }
        
        unordered_set<string> visted;
        for (auto& str : deadends) {
            visted.insert(str);
        }
        if (visted.count(init)) {
            return -1;
        }

        unordered_set<string> st1;
        unordered_set<string> st2;
        st1.insert(init);
        st2.insert(target);
        int step = 0;

        while (!st1.empty() && !st2.empty()) {
            if (st1.size() > st2.size()) {
                swap(st1, st2);
            }

            unordered_set<string> st3;
            step++;
            for (string word : st1) {
                visted.insert(word);
                if (helper(word, visted, st2, st3)) {
                    return step;
                }
            }
            st1 = st3;
        }

        return -1;
    }
};
```

## [剑指 Offer II 110. 所有路径](https://leetcode.cn/problems/bP4bmD/)

**DFS+回溯**

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void dfs(vector<vector<int>>& graph, int i){
        if(i == graph.size() - 1){
            res.push_back(path);
            return;
        }
        for(int k = 0; k < graph[i].size(); k++){
            path.push_back(graph[i][k]);
            dfs(graph,graph[i][k]);
            path.pop_back();
        }
    }
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        path.push_back(0);
        dfs(graph,0);
        return res;
    }
};
```

## [剑指 Offer II 111. 计算除法](https://leetcode.cn/problems/vlzXQL/)

**建图：**

- 利用哈希表存储边与权重，由题意可知，该题为有向图，边的权重与两节点之商

  ```c++
  //map<起点，vector<pair<终点，权重>>>
  unordered_map<string,vector<pair<string,double>>> graph;
  ```

- DFS寻找路径

```c++
class Solution {
public:
    double dfs( unordered_map<string,vector<pair<string,double>>>& graph, string begin, string end,unordered_set<string>& visted, double val){
        if(!graph.count(begin) || !graph.count(end)) return (double)-1;
        visted.insert(begin);
        if(begin == end) return val;
        for(auto s : graph[begin]){
            if(!visted.count(s.first)){
                double ret = dfs(graph,s.first,end,visted,s.second * val);
                if(ret > 0) return ret;
            }
        }
        return -1;
    }

    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        //map<起点，vector<pair<终点，权重>>>
        unordered_map<string,vector<pair<string,double>>> graph;
        //建图
        for(int i = 0; i < equations.size(); i++){
            graph[equations[i][0]].push_back({equations[i][1], values[i]}) ;
            graph[equations[i][1]].push_back({equations[i][0], 1 / values[i]});
        }

        vector<double> res(queries.size(),-1.0);
        for(int i = 0; i < queries.size(); i++){
            unordered_set<string> visted;
            string begin = queries[i][0];
            string end = queries[i][1];
            res[i] = dfs(graph, begin,end, visted, 1);
        }
        return res;
    }
};
```

# 拓扑排序

## [剑指 Offer II 112. 最长递增路径](https://leetcode.cn/problems/fpTFWP/)

**DFS加记忆化搜索**

若使用普通的DFS，则存在大量重复计算，下面阐述普通DFS的算法流程

- 定义`path[i][j]`表示以点`(i,j)`为起点的最长递增路径

- 对于当前点`(i,j)`，遍历其上下左右4个方向上的点，若某方向上的数值比`(i,j)`大，则点`(i,j)`的递增路径长度**加一**
  - **记忆化搜索即是对这一步骤进行优化**

记忆化搜索流程：

- 对于普通DFS的第二步，在以不同点为起点时，存在大量重复过程。若在遍历过程中实时记录以每个点为起点时的最长递增路径，则下次遍历到该点时，可以直接利用该值

**代码：**

```c++
class Solution {
public:
    int dfs(vector<vector<int>>& matrix, int i, int j, int m, int n, vector<vector<int>>& path){
        int ret = 0;
        int dx[] = {0,1,0,-1};
        int dy[] = {1,0,-1,0};
        for(int k = 0; k < 4; k++){
            int a = i + dx[k];
            int b = j + dy[k];
            if(a >= 0 && a < m && b >= 0 && b < n && matrix[a][b] > matrix[i][j]){
                //ret == 以点(a,b)为起点的最长递增路径长度
                if(path[a][b])
                    ret = max(ret, path[a][b]);
                else
                    ret = max(ret, dfs(matrix, a, b, m, n, path));
            }
        }
        //记录以点(i,j)为起点的最长递增路径长度,path[i][j] = max(path[a][b]) + 1,(a,b)为其上下左右4个方向上的点的最长递增路径
        path[i][j] = ret + 1;
        return path[i][j];
    }
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> path(m, vector<int>(n,0));
        int res = 0;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(path[i][j]) 
                    res = max(res, path[i][j]);
                else
                    res = max(res, dfs(matrix, i, j, m, n, path));
            }
        }
        return res;
    }
};
```

**拓扑排序**

- 每个点的出度即有多少条递增边从该单元格出发

```c++
class Solution {
private:
    const int dx[4] = {0,-1,0,1};
    const int dy[4] = {1,0,-1,0};
public:
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return 0;
        }
        int m = matrix.size();
        int n = matrix[0].size();

        //出度数组
        vector<vector<int>> outDegree(m,vector<int>(n,0));
        //统计出度
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                for(int k = 0; k < 4; k++){
                    int a = i + dx[k];
                    int b = j + dy[k];
                    if(a >= 0 && a < m && b >= 0 && b < n && matrix[a][b] > matrix[i][j])
                        outDegree[i][j]++;
                }
            }
        }

        queue<pair<int,int>> q;
        //出度为 0 的点入队列
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(outDegree[i][j] == 0) q.push({i,j});
            }
        }

        int res = 0;
        while(!q.empty()){
            res++;
            int size = q.size();
            for(int i = 0; i < size; i++){
                auto cur = q.front(); q.pop();
                int r = cur.first;
                int c = cur.second;
                for(int k = 0; k < 4; k++){
                    int a = r + dx[k];
                    int b = c + dy[k];

                    if(a >= 0 && a < m && b >= 0 && b < n && matrix[a][b] < matrix[r][c]){
                        --outDegree[a][b];
                        if(outDegree[a][b] == 0) q.push({a,b});
                    }
                }
            }
        } 
        return res;
    }
};
```



## [剑指 Offer II 113. 课程顺序](https://leetcode.cn/problems/QA2IGt/)

**拓扑排序**

`(a,b)`表示课程a必须在课程b之后学习，即拓扑排序后，a必须出现在b的后面,即排序结果为 b a

**使用邻接矩阵建图**

- `graph[i][j]`表示 i 到 j 存在一条边

**拓扑排序流程：**

- 首先将所有入度为0的点加入队列，然后从队头元素开始遍历其临边
- 将临边节点入度减一，若减一后入度为0，则将其加入队列
- 每次将队头元素加入结果集
- 若最后结果集元素数量等于课程数量，说明找到了一条拓扑排序路径，否则不存在拓扑排序

**代码：**

```c++
class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> res;
        //建图,有向图，a->b表示排序结果中 b必定出现在a后面，即拓扑排序后为 a, b
        vector<vector<int>> graph(numCourses,vector<int>(numCourses,0));
        //统计各节点入度
        vector<int> inNum(numCourses,0);
        for(auto p : prerequisites){
            inNum[p[0]]++;
            graph[p[1]][p[0]] = 1;
        }

        queue<int> q;
        //所有入度为0的点入队
        for(int i = 0; i < numCourses; i++){
            if(inNum[i] == 0)  q.push(i);
        }

        while(!q.empty()){
            int cur = q.front();
            res.push_back(cur);
            q.pop();
            for(int i = 0; i < numCourses; i++){
                if(graph[cur][i] == 0) continue;
                inNum[i]--;
                if(inNum[i] == 0){
                    q.push(i);
                }
            }
        }

        if(res.size() == numCourses)
            return res;
        return vector<int>{};

    }
};
```

**优化**

- 是邻接表建图，可以节省大量空间，邻接表用哈希表实现

```c++
class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> res;
        //建图,有向图，a->b表示排序结果中 b必定出现在a后面，即拓扑排序后为 a, b
       unordered_map<int, vector<int>> graph;
        //统计各节点入度
        vector<int> inNum(numCourses,0);
        for(auto p : prerequisites){
            inNum[p[0]]++;
            graph[p[1]].push_back(p[0]);
        }

        queue<int> q;
        //所有入度为0的点入队
        for(int i = 0; i < numCourses; i++){
            if(inNum[i] == 0)  q.push(i);
        }

        while(!q.empty()){
            int cur = q.front();
            res.push_back(cur);
            q.pop();
            for(auto& n : graph[cur]){
                inNum[n]--;
                if(inNum[n] == 0){
                    q.push(n);
                }
            }
        }

        if(res.size() == numCourses)
            return res;
        return{};

    }
};
```

## [剑指 Offer II 114. 外星文字典](https://leetcode.cn/problems/Jf1JuT/)

对字符串 s1 和字符串 s2 进行排序时，从两者的首字母开始逐位对比，先出现较小的字母的单词排序在前，若直到其中一个单词所有字母都对比完都无法得出结果，则长度较小的单词排序在前。以题目中 words = ["ac","ab","bc","zc","zb"] 为例，一共出现了 'a'、'b'、'c' 和 'z' 4 个字母。由于 "ac" < "ab" 所以 'c' < 'b'，由于 "ab" < "bc" 所以 'a' < 'b'，依次可得 'b' < 'z' 、'c' < 'b'。

这里需要注意一类特殊输入，若两者的前缀相同，但前者的单词长度长于后者，如 "abc" 和 "ab"。这是不符合排序规则的，无论最后字母存在怎么样的拓扑排序都不会成立，所以这是一个无效输入，直接输出拓扑排序为空。

```c++
class Solution {
public:
    string alienOrder(vector<string>& words) {
        unordered_map<char, unordered_set<char>> graph;
        unordered_map<char, int> inDegress;

        // 建图
        for (auto& word : words) {
            for (auto& ch : word) {
                if (!graph.count(ch)) {
                    graph[ch] = {};
                } 
                if (!inDegress.count(ch)) {
                    inDegress[ch] = 0;
                }
            }
        }

        // 计算邻接表和入度
        for (int i = 1; i < words.size(); ++i) {
            int j = 0;
            int len = min(words[i - 1].size(), words[i].size());
            for (; j < len; ++j) {
                char ch1 = words[i - 1][j];
                char ch2 = words[i][j];
                //若ch1 != ch2 说明字典序中ch1 < ch2，建立一条边：ch1->ch2
                if (ch1 != ch2) {
                    if (!graph[ch1].count(ch2)) {
                        graph[ch1].insert(ch2);
                        inDegress[ch2]++;
                    }
                    break;
                }
            }
            // 特殊判断
            if (j == len && words[i - 1].size() > words[i].size()) {
                return {};
            }
        }

        string ret{""};
        queue<char> que;
        // 入度为 0 的点
        for (auto& d : inDegress) {
            if (d.second == 0) {
                que.push(d.first);
            }
        }
        // BFS
        while (!que.empty()) {
            char ch = que.front();
            que.pop();
            ret.push_back(ch);

            for (auto& c : graph[ch]) {
                inDegress[c]--;
                if (inDegress[c] == 0) {
                    que.push(c);
                }
            }
        }
        
        if (ret.size() != inDegress.size()) {
            return "";
        }
        return ret;
    }
};
```

## [剑指 Offer II 115. 重建序列](https://leetcode.cn/problems/ur2n8P/)

**妈的这题贼烦，细节很多**

**首先，这题是典型的拓扑排序题，首先明确题意：**

- 该题是求：给定序列`org`是否是**唯一**的拓扑排序，而不是单纯的求是否存在拓扑排序

**注意细节：**

- 元素是`1~n`，因此入度数组大小应该为`org.size() + 1`,从元素1开始计算入度
- 对于给定的序列集`seqs`中的每个集合，要判断其合法性，若集合中的元素小于1或者大于`org.size()`，则说明该元素不合法，直接返回false
- 若要拓扑排序唯一，则每个节点的入度都应该为一，即BFS每一层循环，队列中都只有一个元素
- 最后判断构建出的拓扑排序序列是否与给定序列相同

**代码：**

```c++
class Solution {
public:
    bool sequenceReconstruction(vector<int>& org, vector<vector<int>>& seqs) {
        if(seqs.empty()) return false;
        unordered_map<int,vector<int>> graph;
        vector<int> inDegree(org.size() + 1,0);
        vector<int> ret;
        //建图
        for(auto v : seqs){
            //判断合法性
            if(v.size() == 1 && (v[0] > org.size() || v[0] < 1)){
                return false;
            }
            for(int i = 1; i < v.size(); i++){
                //判断合法性
                if(v[i] > org.size() || v[i] < 1) return false;
                graph[v[i-1]].push_back(v[i]);
                inDegree[v[i]]++;
            }
        }

        queue<int> q;
        for(int i = 1; i < inDegree.size(); i++){
            if(inDegree[i] == 0)
                q.push(i);
        }

        while(!q.empty()){
            int size = q.size();
            //若要拓扑排序唯一，则每个节点的入度都应该为一，即BFS每一层循环，队列中都只有一个元素
            if(size != 1) return false;
            int cur = q.front();
            q.pop();
            ret.push_back(cur);
            for(auto g : graph[cur]){
                inDegree[g]--;
                if(inDegree[g] == 0)
                    q.push(g);
            }
        }
		//若构建出的拓扑排序序列与给定的相同，返回true
        return ret == org;
    }
};
```

# 并查集

## [剑指 Offer II 106. 二分图](https://leetcode.cn/problems/vEAB3K/)

```c++
class Solution {
public:
    int p[101];
    int find(int x){
        if(p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    void merge(int i, int j){
        int p1 = find(i);
        int p2 = find(j); 
        if(p1 != p2){
            p[p1] = p2;
        }
    }
    bool isBipartite(vector<vector<int>>& graph) {
        int n = graph.size();
        for(int i = 0; i < n; i++){
            p[i] = i;
        }
        for(int i = 0; i < n; i++){
            for(int j = 0; j < graph[i].size(); j++){
                int p1 = find(i);
                int p2 = find(graph[i][j]);
                if(p1 == p2) return false;
                merge(graph[i][0], graph[i][j]);
            }
        }
        return true;
    }
};
```

## [剑指 Offer II 111. 计算除法](https://leetcode.cn/problems/vlzXQL/)

### 并查集

```c++
class Solution {
public:
    int findf(vector<int>& f, vector<double>& w, int x) {
        if (f[x] != x) {
            int father = findf(f, w, f[x]);
            w[x] = w[x] * w[f[x]];
            f[x] = father;
        }
        return f[x];
    }

    void merge(vector<int>& f, vector<double>& w, int x, int y, double val) {
        int fx = findf(f, w, x);
        int fy = findf(f, w, y);
        f[fx] = fy;
        w[fx] = val * w[y] / w[x];
    }

    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        int nvars = 0;
        unordered_map<string, int> variables;

        int n = equations.size();
        //计算存在多少个不同字符
        for (int i = 0; i < n; i++) {
            if (variables.find(equations[i][0]) == variables.end()) {
                variables[equations[i][0]] = nvars++;
            }
            if (variables.find(equations[i][1]) == variables.end()) {
                variables[equations[i][1]] = nvars++;
            }
        }
        vector<int> f(nvars);
        vector<double> w(nvars, 1.0);
        for (int i = 0; i < nvars; i++) {
            f[i] = i;
        }

        for (int i = 0; i < n; i++) {
            int va = variables[equations[i][0]], vb = variables[equations[i][1]];
            merge(f, w, va, vb, values[i]);
        }
        vector<double> ret;
        for (const auto& q: queries) {
            double result = -1.0;
            if (variables.find(q[0]) != variables.end() && variables.find(q[1]) != variables.end()) {
                int ia = variables[q[0]], ib = variables[q[1]];
                int fa = findf(f, w, ia), fb = findf(f, w, ib);
                if (fa == fb) {
                    result = w[ia] / w[ib];
                }
            }
            ret.push_back(result);
        }
        return ret;
    }
};
```



## [剑指 Offer II 116. 省份数量](https://leetcode.cn/problems/bLyHh0/)

### BFS

计算图中子图的数量，使用BFS，遍历每个节点以及其子节点，并标记为已访问，每进行一次BFS就找到了一个子图，统计BFS的次数即为答案

```c++
class Solution {
public:
    void Bfs(int i, vector<vector<int>>& isConnected, vector<bool>& visit){
        queue<int> q;
        q.push(i);
        visit[i] = true;
        while(!q.empty()){
            int cur = q.front();
            q.pop();
            for(int j = 0; j < isConnected.size(); j++){
                if(isConnected[cur][j] && !visit[j]){
                    visit[j] = true;
                    q.push(j);
                }
            }
        }
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        vector<bool> visit(n,false);
        int res = 0;
        for(int i = 0; i < isConnected.size(); i++){
            if(!visit[i]){
                Bfs(i,isConnected,visit);
                res++;
            }
        }
        return res;
    }
};
```

### **并查集**

```c++
class Solution {
public:
    //1 <= n <= 200，因此只需要开辟200
    int p[200];
    //返回x的父节点
    int find(int x){
        if(p[x] != x) p[x] = find(p[x]);
        return p[x];
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        //初始化所有节点的父节点为自己
        for(int i = 0; i < 200; i++)
            p[i] = i;

        for(int i = 0; i < isConnected.size(); i++){
            for(int j = 0; j < isConnected.size(); j++){
                //若两节点联通且不在同一集合，则合并集合
                if(i != j && isConnected[i][j]){
                    int p1 = find(i);
                    int p2 = find(j);
                    //若两者不属于同一个集合
                    if(p1 != p2){
                        p[p1] = p2;
                    }
                }
            }
        }
        //并查集中父节点的个数就是省份数量
        int count = 0;
        for(int i = 0; i < isConnected.size(); i++){
            if(p[i] == i) count++;
        }
        return count;
    }
};
```



## [剑指 Offer II 117. 相似的字符串](https://leetcode.cn/problems/H6lPxb/)

### BFS

```c++
class Solution {
public:
    //因为给定字符串全互为字母异位词，因此只需要判断是否只有两个位置不同或全相同即可判断两者是否相似
    bool isSimilar(string s1, string s2){
        int cnt = 0;
        for(int i = 0; i < s1.length(); ++i){
            if(s1[i] != s2[i]) cnt++;
        }
        return cnt <= 2;
    }

    void bfs(vector<string>& strs,  vector<bool>& vis, int i, unordered_map<string,unordered_set<string>>& mp){
        queue<int> q;
        q.push(i);
        vis[i] = true;
        while(!q.empty()){
            int cur = q.front();
            q.pop();
            for(int j = 0; j < strs.size(); j++){
                if(!vis[j] && mp[strs[cur]].count(strs[j])){
                    q.push(j);
                    vis[j] = true;
                }
            }
        }
    }

    int numSimilarGroups(vector<string>& strs) {
        int n = strs.size();
        int cnt = 0;
        vector<bool> vis(n,false);
        //建图
        unordered_map<string,unordered_set<string>> mp;
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < n; ++j){
                if(isSimilar(strs[i], strs[j])){
                    mp[strs[i]].insert(strs[j]);
                }
            }
        }
        for(int i = 0; i < n; ++i){
            if(!vis[i]){
                bfs(strs, vis, i, mp);
                cnt++;
            }
        }
        return cnt;
    }
};
```

### 并查集

```c++
class Solution {
public:
    vector<int> f;

    int find(int x) {
        return f[x] == x ? x : f[x] = find(f[x]);
    }

    bool check(const string &a, const string &b, int len) {
        int num = 0;
        for (int i = 0; i < len; i++) {
            if (a[i] != b[i]) {
                num++;
                if (num > 2) {
                    return false;
                }
            }
        }
        return true;
    }

    int numSimilarGroups(vector<string> &strs) {
        int n = strs.size();
        int m = strs[0].length();
        f.resize(n);
        for (int i = 0; i < n; i++) {
            f[i] = i;
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int fi = find(i), fj = find(j);
                if (fi == fj) {
                    continue;
                }
                if (check(strs[i], strs[j], m)) {
                    f[fi] = fj;
                }
            }
        }
        int ret = 0;
        for (int i = 0; i < n; i++) {
            if (f[i] == i) {
                ret++;
            }
        }
        return ret;
    }
};

```

## [剑指 Offer II 118. 多余的边](https://leetcode.cn/problems/7LpjUW/)

可以通过并查集寻找多余的边。初始时，每个节点都属于不同的连通分量。遍历每一条边，判断这条边连接的两个顶点是否属于相同的连通分量。

- 如果两个顶点属于不同的连通分量，则说明在遍历到当前的边之前，这两个顶点之间不连通，因此当前的边不会导致环出现，合并这两个顶点的连通分量。


- 如果两个顶点属于相同的连通分量，则说明在遍历到当前的边之前，这两个顶点之间已经连通，因此当前的边导致环出现，为多余的边，将当前的边作为答案返回。


```c++
class Solution {
public:
    int p[1001];

    int find(int x) {
        if(p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        for (int i = 1; i < edges.size(); ++i) {
            p[i] = i;
        }

        int ret = -1;
        for (int i = 0; i < edges.size(); ++i) {
            int p1 = find(edges[i][0]);
            int p2 = find(edges[i][1]);
            if (p1 != p2) {
                p[p2] = p1;
            }
            else {
                ret = i;
            }
        }
        
        if (ret == -1) {
            return {};
        }
        return edges[ret];
    }
};

```

## [剑指 Offer II 119. 最长连续序列](https://leetcode.cn/problems/WhsWhI/)

### BFS 

如果把每个整数看作图的节点，相差为 1 的两个整数之间存在一条边，那么这些整数就会形成若干个子图，每个子图内都对于一个连续数值序列，那么本问题就转化为求最大的子图大小。可以使用图的广度优先搜索和深度优先搜索两种算法计算每个子图，这里采用广度优先算法。完整代码如下，若整数的个数为 n，那么节点数为 n，边的个数为 O(n)，所以算法的时间复杂度为 O(n)。

```c++
class Solution {
private:
    int bfs(unordered_set<int>& visted, int node) {
        queue<int> que;
        que.push(node);

        int len = 0;
        while (!que.empty()) {
            int cur = que.front();
            que.pop();
            visted.erase(cur);
            len++;
            if (visted.count(cur + 1)) {
                que.push(cur + 1);
            }
            if (visted.count(cur - 1)) {
                que.push(cur - 1);
            }
        }
        return len;
    }

public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> visted;
        for (auto& n : nums) {
            if (!visted.count(n)) {
                visted.insert(n);
            }
        }

        int ret = 0;
        for (auto& n : nums) {
            if (visted.count(n)) {
                ret = max(ret, bfs(visted, n));
            }
        }

        return ret;
    }
};
```

### 并查集

**数组中可能存在重复元素，我写的这版代码过不去**

```c++
class Solution {
public:
    int p[10001];
    //连通块大小，
    int size[10001];
    int find(int x){
        if(p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    int longestConsecutive(vector<int>& nums) {
        for(int i = 0; i < nums.size(); i++){
            p[i] = i;
            size[i] = 1;
        }

        for(int i = 0; i < nums.size(); i++){
            for(int j = 0; j < nums.size(); j++){
                int p1 = find(i);
                int p2 = find(j);
                if(p1 != p2 && abs(nums[i] - nums[j]) == 1){
                    p[p1] = p2;
                    size[p2] += size[p1];
                }
            }
        }

        int res = 0;
        for(int i = 0; i < nums.size(); i++){
            res = max(res, size[i]);
        }
        return res;
    }
};
```

**因此需要使用哈希表来模拟并查集**

```c++
class Solution {
private:
    unordered_map<int, int> p;
    unordered_map<int, int> counts;
    
    int find(int x){
        if(p[x] != x) p[x] = find(p[x]);
        return p[x];
    }

    int merge(int node1, int node2) {
        int p1 = find(node1);
        if (p.count(node2)) {
            int p2 = find(node2);
            if (p1 != p2) {
                p[p2] = p1;
                counts[p1] += counts[p2];
            }
        }
        return counts[p1];
    }

public:
    int longestConsecutive(vector<int>& nums) {
        for (auto& n : nums) {
        	p[n] = n;
        	counts[n] = 1;
        }

        int ret = 0;
        for (auto& n : nums) {
            ret = max(ret, merge(n, n - 1));
            ret = max(ret, merge(n, n + 1));
        }

        return ret;
    }
};
```



### 哈希表

首先所有元素存入hashset；
然后遍历哈希表，若发现比当前元素小1的元素存在于哈希表中，说明此元素不是开头，则跳过。
找到开头元素，一直递增（每次+1）到hashset中不包含为止。 此为连续序列。
遍历过程中记录参数即可。

```c++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> hash;
        int res = 0;
        for(const auto num : nums){
            hash.insert(num);
        }
        for(const auto num : hash){
            int len = 1;
            if(hash.count(num - 1)){
                continue;
            }
            int cur = num;
            while(hash.count(cur + 1)){
                cur++;
                len++;
            }
            res = max(res,len);
        }
        return res;
    }
};
```

## [1631. 最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/)

- 计算所有边的权重，权重即为边两端点的**高度差绝对值**
- 将所有边的权重按从小到大排序，**按权重从小到大**合并点
- 若加入某权重为`a`的边后，点`(0,0)`与`(m-1,n-1)`处于同一集合，则说明存在一条路径，将两点联通，并且该路径体力消耗值最小，为`a`

```c++
class UF {
public:
    vector<int> fa;
    vector<int> sz;
    int n;
    //连通块数量
    int comp_cnt;
    
public:
    UF(int _n): n(_n), comp_cnt(_n), fa(_n), sz(_n, 1) {
        iota(fa.begin(), fa.end(), 0);
    }
    
    int findset(int x) {
        return fa[x] == x ? x : fa[x] = findset(fa[x]);
    }
    
    void unite(int x, int y) {
        x = findset(x);
        y = findset(y);
        if (x == y) {
            return;
        }
        if (sz[x] < sz[y]) {
            swap(x, y);
        }
        fa[y] = x;
        sz[x] += sz[y];
        --comp_cnt;
    }
    
    bool connected(int x, int y) {
        x = findset(x);
        y = findset(y);
        return x == y;
    }
};

//自定义边结构
struct Edge {
    int x, y, z;
    Edge(int _x, int _y, int _z): x(_x), y(_y), z(_z) {}
    bool operator< (const Edge& that) const {
        return z < that.z;
    }
};

class Solution {
public:
    int minimumEffortPath(vector<vector<int>>& heights) {
        int m = heights.size();
        int n = heights[0].size();
        vector<Edge> edges;
        //将所有的边加入集合
        //从左向右、从上到下遍历，对于当前点，将其与上方的点以及下方的点的边加进来，遍历所有点以后，图中所有的边也都加入了集合
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                //当前点可由其上方点或者左边点到达
                int id = i * n + j;
                if (i > 0) {
                    //点(i-1,j) 与 (i,j)的高度差绝对值
                    edges.emplace_back(id - n, id, abs(heights[i][j] - heights[i - 1][j]));
                }
                if (j > 0) {
                    //点(i,j-1) 与 (i,j)的高度差绝对值
                    edges.emplace_back(id - 1, id, abs(heights[i][j] - heights[i][j - 1]));
                }
            }
        }
        
        sort(edges.begin(), edges.end());
        UF uf(m * n);
        for (const auto& edge: edges) {
            uf.unite(edge.x, edge.y);
            if (uf.connected(0, m * n - 1)) {
                return edge.z;
            }
        }
        return 0;
    }
};
```



# Dijkstra算法

## [743. 网络延迟时间](https://leetcode.cn/problems/network-delay-time/)

**邻接表**

```c++
class Solution {
public:

    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<pair<int,int>>> graph(n+1);
        //j
        for(auto t : times){
            graph[t[0]].push_back({t[1], t[2]});
        }
        
        vector<int> dist(n+1,INT_MAX);
        dist[k] = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> q;
        q.push({0,k});

        while(!q.empty()){
            auto cur = q.top();q.pop();
            int dis = cur.first;
            int index = cur.second;
            if(dist[index] < dis) continue;

            for(auto g : graph[index]){
                int y = g.first, d = dist[index] + g.second;
                if(d < dist[y]){
                    dist[y] = d;
                    q.push({d,y});
                }
            }
        }

        int res = INT_MIN;
        for(int i = 1; i <= n; i++){
            res = max(res, dist[i]);
        }
        if(res == INT_MAX) return -1;
        return res;
    }
};
```

**邻接矩阵**

```c++
class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<int>> graph(n+1,vector<int>(n+1,INT_MAX));
        vector<int> dist(n+1,INT_MAX);
        dist[k] = 0;

        for(int i = 0;i < times.size(); i++){
            graph[times[i][0]][times[i][1]] = times[i][2];
        }

        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<>> q;
        q.push({k,0});
        while(!q.empty()){
            auto cur = q.top(); q.pop();
            int node = cur.first;
            int time = cur.second;
            for(int i = 0; i < graph[node].size(); i++){
                if(graph[node][i] == INT_MAX) continue;
                if(dist[node] + graph[node][i] < dist[i]){
                    dist[i] = dist[node] + graph[node][i];
                    q.push({i,dist[i]});
                }
            }
        }

        int res = 0;
        for(int i = 1; i <= n; i++){
            res = max(res,dist[i]);
        }
        
        return res == INT_MAX ? -1 : res;
    }
};
```



# Floyd算法

## [1334. 阈值距离内邻居最少的城市](https://leetcode.cn/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)

```c++
class Solution {
public:
    int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold) {
        vector<vector<int>> dist(n,vector<int>(n,INT_MAX));
        for(int i = 0; i < n; i++) dist[i][i] = 0;
        for(auto edge : edges){
            dist[edge[0]][edge[1]] = edge[2];
            dist[edge[1]][edge[0]] = edge[2];
        }

        for(int k = 0; k < n; k++){
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    long ik = dist[i][k], kj = dist[k][j];
                    if(dist[i][j] > ik + kj)
                        dist[i][j] = ik + kj;
                }
            }
        }

        int res = 0,minCount = n;
        for(int i = 0; i < n; i++){
            int count = 0;
            for(int j = 0; j < n; j++){
                if(j == i) continue;
                if(dist[i][j] <= distanceThreshold) count++;
            }
            if(minCount >= count){
                minCount = count;
                res = i;
            }
        }
        return res;
    }
};
```

