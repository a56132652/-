# [剑指 Offer II 001. 整数除法](https://leetcode-cn.com/problems/xoh6Oh/)

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



# [剑指 Offer II 002. 二进制加法](https://leetcode-cn.com/problems/JFETK5/)

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



# [剑指 Offer II 003. 前 n 个数字二进制中 1 的个数](https://leetcode-cn.com/problems/w3tCBm/)

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



# [剑指 Offer II 004. 只出现一次的数字 ](https://leetcode-cn.com/problems/WGki4K/)

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



# [剑指 Offer II 005. 单词长度的最大乘积](https://leetcode-cn.com/problems/aseY1I/)

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

