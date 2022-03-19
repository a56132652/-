# 一些细节问题

## 1）ACM 模式 C++万能头文件`<bits/stdc++.h>`

## 2）二叉树节点

```c++
struct TreeNode{
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode(): val(0), left(nullptr), right(nullptr){}
	TreeNode(int x, TreeNode* l, TreeNode* r) : val(x) , left(l), right(r) {}
};
```



# 1. 平分物品

现在有n个物品，每一个物品都有一个价值，现在想将这些物品分给两个人，要求这两个人每一个人分到的物品的价值总和相同（个数可以不同，总价值相同即可），剩下的物品就需要扔掉，现在想知道最少需要扔多少价值的物品才能满足要求分给两个人。

**输入描述:**

```
第一行输入一个整数 T，代表有 T 组测试数据。
对于每一组测试数据，一行输入一个整数 n ，代表物品的个数。接下来 n 个数，a[i] 代表每一个物品的价值。1<= T <= 101 <= n <= 151 <= a[i] <= 100000
```



**输出描述:**

```
对于每一组测试数据，输出一个答案代表最少需要扔的价值。
```



**输入例子1:**

```
1
5
30 60 5 15 30
```



**输出例子1:**

```
20
```



**例子说明1:**

```
样例解释，扔掉第三个和第四个物品，然后将第一个物品和第五个物品给第一个人，第二个物品给第二个人，每一个人分到的价值为，扔掉的价值为。   
```

## **思路：**

两人从 0 开始拿，对于每一件物品开始进行选择，对于每个物品有三种选择，给第一个人、给第二个人、丢掉。

## 代码：

**别人的代码**

```c++
#include<iostream>
using namespace std;
#include<vector>
#include<algorithm>
#include<bits/stdc++.h>
//定义两组数据的累加和  从0开始dfs每种可能，遇到两者相同的情况，就记录此时需要扔掉
//选择问题  两个人从0开始拿物品，遇到一个物品有三种选择，给第一个人，给第二个人，扔掉。
//走到结尾就找到舍弃价值最小的那一个节点
  
int res = INT_MAX;//最小扔掉的价值

void dfs(vector<int>& nums,int result1,int result2,int sum,int index,int n)
{
    //一直选择到最后一个数字才返回
    if (index == n)
    {
        if (result1 == result2)
        {
            res = min(res, sum - result1 - result2);
        }
        return;
    }
     
    //选择环节  每次进入选择环节都有三种选择 
    dfs(nums,result1 + nums[index], result2, sum, index + 1,n);
    dfs(nums,result1,result2 + nums[index], sum, index + 1,n);
    dfs(nums,result1,result2,sum,index+1,n);
}
  
int main()
{
    int t;
    cin >> t;
    while (t--)//一个while输出一个答案
    {
        int n;
        cin >> n;
        int temp;
          
        vector<int> nums;//输入数组
        for (int i = 0; i < n; i++)
        {
            cin >> temp;
            nums.push_back(temp);
        }
  
        int sum = 0;
        for (auto i : nums)
        {
            sum += i;
        }
  
        dfs(nums, 0, 0, sum, 0, n);
        cout << res << endl;
        res = INT_MAX;
  
    }
}
```



```c++
#include<iostream>
using namespace std;
const int N = 1e5+10;
int a[N];

int res = 1e9;
void dfs(int result1,int result2,int sum,int n, int i) {
    if(i==n) {
        if(result1==result2) {
            res = min(res, sum-result1-result2);
        }
        return;
    }
    dfs(result1+ a[i],result2,sum,n,i+1);
    dfs(result1,result2+a[i],sum,n,i+1);
    dfs(result1,result2,sum,n,i+1);
    
}

int main() {
    int t;scanf("%d",&t);
    
    while(t--) {
        int n;scanf("%d", &n);
        res = 1e9;
        int sum = 0;
        for(int i=0;i<n;++i) scanf("%d", &a[i]),sum+=a[i];
        dfs(0,0,sum,n,0);
        printf("%d\n",res);
    }
    
}

```





# 2. 买票问题

现在有n个人排队买票，已知是早上8点开始卖票，这n个人买票有两种方式：

第一种是每一个人都可以单独去买自己的票，第 i 个人花费 a[i] 秒。

第二种是每一个人都可以选择和自己后面的人一起买票，第 i 个人和第 i+1 个人一共花费 b[i] 秒。

最后一个人只能和前面的人一起买票或单独买票。

由于卖票的地方想早些关门，所以他想知道他最早几点可以关门，请输出一个时间格式形如：08:00:40 am/pm

时间的数字要保持 2 位，若是上午结束，是 am ，下午结束是 pm

**输入描述:**

```
第一行输入一个整数 T，接下来有 T 组测试数据。对于每一组测试数据：输入一个数 n，代表有 n 个人买票。接下来n个数，代表每一个人单独买票的时间 a[i]。
接下来 n-1 个数，代表每一个人和他前面那个人一起买票需要的时间 b[i]
1<= T <=100
1<= n <=2000
1<= a[i] <=50
1<= b[i] <=50
```



**输出描述:**

```
对于每组数据，输出一个时间，代表关门的时间 。
```



**输入例子1:**

```
2
2
20 25
40
1
8
```



**输出例子1:**

```
08:00:40 am
08:00:08 am
```



## **思路:**

**动态规划：**

- `dp[i]`: 前 `i`个人买票所花的最小时间
- 状态转移方程：对于每个人，有两种状态，一是自己单独买，二是与前一个人一起买
  - 情况一：`dp[i] = dp[i-1] + single[i]`
  - 情况二：`dp[i] = dp[i-2] + twoSum`
  - 取两者最小值，因此`dp[i] = min(dp[i-1] + single[i], dp[i-2] + twoSum)`
- 初始条件：
  - 前第一个人只能单独买（当只有一个人时），因此`dp[1] = a[0]`

**注意输出格式，这里采用**`%02d`形式输出，具体解释如下：

- `%d`:就是普通的输出了，%d 是输出十进制整数
- `%2d`:将数字按宽度为2，采用右对齐方式输出，如果数据位数不到2位，则左边补空格

- **`%02d`**:默认情况下，数据数据宽度不够2位是用空格填补的，但是因为`2d`前面有`0`，表示，数据宽度不足时用`0`补

```c++
#include <bits/stdc++.h>
  
using namespace std;
  
int main() {
    int T;
    cin >> T;
    while (T--) {
        int n;
        cin >> n;
        vector<int> dp(n+1 , 0);
        vector<int> single(n);
        for (int i = 0; i < n; ++i) {
            cin >> single[i];
        }
        dp[1] = single[0];
        for (int i = 2; i <= n; ++i) {
            int twoSum;
            cin >> twoSum;
            dp[i] = min(dp[i - 1] + single[i-1], dp[i - 2] + twoSum);
        }
          
        int seconds = dp[n];
        int hh, mm, ss;
        ss = seconds % 60;
        mm = (seconds / 60) % 60;
        hh = seconds / 3600;
        hh += 8;
        char s[3] = "am";
        if (hh > 12) {
            hh -= 12;
            s[0] = 'p';
        }
        printf("%02d:%02d:%02d %s\n", hh, mm, ss, s);
          
    }
    return 0;
}
```



# 3. 小易爱回文

时间限制：C/C++ 1秒，其他语言2秒

空间限制：C/C++ 256M，其他语言512M

小易得到了一个仅包含大小写英文字符的字符串，该字符串可能不是回文串。（“回文串”是一个正读和反读都一样的字符串，比如“level”或者“noon”等等就是回文串，“asds”就不是回文串。）

小易可以在字符串尾部加入任意数量的任意字符，使其字符串变成回文串。

现在请你编写一个程序，程序要能计算出小易可以得到的最短回文串。



**输入描述:**

```
一行包括一个字符串。
```



**输出描述:**

```
一行包括一个字符串，代表答案。
```



**输入例子1:**

```
noon
```



**输出例子1:**

```
noon
```



**输入例子2:**

```
noo
```

## 

## 思路：

从给定字符串的首字母`s[i]`开始遍历，

- 若从`s[i]`开始的子串不是回文串，则将`s[i]`记录进一个临时字符串`temp`的头部
- 若从`s[i]`开始的子串是回文串，则将给定字符串`s`与`temp`拼接，作为答案返回

**例如：给定 `s = noo`**

1. `noo`不是回文串， `temp = n`
2. `oo`是回文串，返回`s + temp = noon`

**例如：给定 `s = hello`**

1. `hello 不是回文串，temp = h`
2. `ello 不是回文串，temp = eh`
3. `llo 不是回文串，temp = leh`
4. `lo 不是回文串，temp = lleh`
5. `o 是回文串，拼接，返回 hellolleh`

## 代码：

```c++
#include<bits/stdc++.h>
using namespace std;
 
bool isP(string s, int start){
     for(int i = start, j = s.size()-1; i<=(i+j)/2; i++, j--){
            if(s[i]!=s[j]){
 
                return false;
            }
      }
 
    return true;
}
int main(){
 
    string s;
 
    string ans;
    int flag = 0;
    while(cin>>s){
        ans.clear();
 
        for(int i =0; i< s.size();i++){
            if(!isP(s,i)){//如果不是，那这个字符倒着加到新字符串得第一位
               flag =1;
               ans.insert(ans.begin(),s[i]);
            }else{
                break;//如果从下一个字母开始找是回文串，直接返回。就可以涵盖原先是回文得情况
            }
        }
 
        if(flag==1){
            cout<<s+ans<<endl;
        }else{
            cout<<s<<endl;
        }
    }
    return 0;
}
```





# 4. 淘汰分数

时间限制：C/C++ 1秒，其他语言2秒

空间限制：C/C++ 256M，其他语言512M

某比赛已经进入了淘汰赛阶段,已知共有n名选手参与了此阶段比赛，他们的得分分别是a_1,a_2….a_n,小美作为比赛的裁判希望设定一个分数线m，使得所有分数大于m的选手晋级，其他人淘汰。

但是为了保护粉丝脆弱的心脏，小美希望晋级和淘汰的人数均在[x,y]之间。

显然这个m有可能是不存在的，也有可能存在多个m，如果不存在，请你输出-1，如果存在多个，请你输出符合条件的最低的分数线。

**输入描述:**

```
输入第一行仅包含三个正整数n,x,y，分别表示参赛的人数和晋级淘汰人数区间。(1<=n<=50000,1<=x,y<=n)输入第二行包含n个整数，中间用空格隔开，表示从1号选手到n号选手的成绩。(1<=|a_i|<=1000)
```



**输出描述:**

```
输出仅包含一个整数，如果不存在这样的m，则输出-1，否则输出符合条件的最小的值。
```



**输入例子1:**

```
6 2 3
1 2 3 4 5 6
```



**输出例子1:**

```
3
```

## 思路：

其实这题可以利用快速排序的性质，每一轮快排以后，都将flag 与 n[left]进行了交换，此时，在 flag 的左边都是比他小的数，右边都是比他大的数，因此可以通过判断 flag左右两边的数字的数量是否满足 [x,y]区间 ，来找出分数线，

我这里是直接利用了库函数，因为库函数底层也是快排，然后再顺序遍历，时间复杂度其实也是 nlogn

## 代码：

```c++
#include<bits/stdc++.h>
#include<algorithm>
using namespace std;


int main()
{
    int n, x, y;
    cin >> n >> x >> y;
    if(x > y){
        cout << -1 << endl;
        return 0;
    }
    vector<int> a(n);
    for(int i = 0; i < n; i++) cin >> a[i];
    sort(a.begin(),a.end());
    int out = x, in = n - x;
    for(int i = x - 1; i < n - x; i++)
    {
        if(out >= x && out <= y && in >= x && in <= y){
            cout << a[i] << endl;
            return 0;
        }
        else{
            out++;
            in--;
        }
    }
    cout << -1 << endl;
    return 0;
}
```



# 5. 正则序列

时间限制：C/C++ 1秒，其他语言2秒

空间限制：C/C++ 256M，其他语言512M

我们称一个长度为n的序列为正则序列，当且仅当该序列是一个由1~n组成的排列，即该序列由n个正整数组成，取值在[1,n]范围，且不存在重复的数，同时正则序列不要求排序

有一天小团得到了一个长度为n的任意序列s，他需要在有限次操作内，将这个序列变成一个正则序列，每次操作他可以任选序列中的一个数字，并将该数字加一或者减一。

请问他最少用多少次操作可以把这个序列变成正则序列？

**输入描述:**

```
输入第一行仅包含一个正整数n，表示任意序列的长度。(1<=n<=20000)输入第二行包含n个整数，表示给出的序列，每个数的绝对值都小于10000。
```



**输出描述:**

```
输出仅包含一个整数，表示最少的操作数量。
```

**输入例子1:**

```
5
-1 2 3 10 100
```

**输出例子1:**

```
103
```



## 思路：

先排序，然后一个个变q

## 代码：

```c++
#include <bits/stdc++.h>
using namespace std;

int main()
{
    int n;
    cin >> n;
    vector<int> nums(n);
    for(int i = 0; i < n; i++) cin >> nums[i];
    sort(nums.begin(),nums.end());
    int res = 0;
    int cur = 1;
    for(auto num : nums){
        res += cur - num;
        cur++;
    }
    cout << res << endl;
    return 0;
}
```



