# [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```c++
class Solution {
private:
    const int dx[4] = {0,-1,0,1};
    const int dy[4] = {1,0,-1,0};

    void dfs(vector<vector<char>>& grid,int i, int j, int n, int m)
    {
        if(i < 0 || i >= n || j < 0 || j >= m || grid[i][j] != '1') return;

        int cnt = 1;
        grid[i][j] = '2';
        for(int k = 0; k < 4; k++)
        {
            int a = i + dx[k];
            int b = j + dy[k];

            dfs(grid, a, b,n,m);
        }
    }

public:
    int numIslands(vector<vector<char>>& grid) {
        int nr = grid.size();
        if (!nr) return 0;
        int nc = grid[0].size();

        int num_islands = 0;
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    dfs(grid, r, c,nr,nc);
                }
            }
        }

        return num_islands;
    }
};
```





# [463. 岛屿的周长](https://leetcode-cn.com/problems/island-perimeter/)

```c++
class Solution {
public:
    const int dx[4] = {0,-1,0,1};
    const int dy[4] = {1,0,-1,0};

    int dfs(vector<vector<int>>& grid, int i, int j, int n, int m)
    {
        int ans = 0;
        if(i < 0 || i >= n || j < 0 || j >= m || grid[i][j] == 0) return 1;
        if(grid[i][j] == 2) return 0;

        grid[i][j] = 2;
        for(int k = 0; k < 4; k++)
        {
            int a = i + dx[k];
            int b = j + dy[k];
            ans += dfs(grid,a,b,n,m);
        }
        return ans;
    }

    int islandPerimeter(vector<vector<int>>& grid) {
        if(grid.empty() || grid[0].empty()) return 0;
        int n = grid.size(), m = grid[0].size();
        int res = 0;
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                if(grid[i][j] == 1)
                    res += dfs(grid,i,j,n,m);
            }
        }
        return res;
    }
};
```



# [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

```c++
class Solution {
public:
    const int dx[4] = {0,-1,0,1};
    const int dy[4] = {1,0,-1,0};

    int dfs(vector<vector<int>>& grid,int i, int j, int n, int m)
    {
        if(i < 0 || i >= n || j < 0 || j >= m || grid[i][j] != 1) return 0;

        int area = 1;
        grid[i][j] = 2;
        for(int k = 0; k < 4; k++)
        {
            int a = i + dx[k];
            int b = j + dy[k];

            area += dfs(grid,a,b,n,m);
        }
        return area;
    }
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int n = grid.size();
        int m = grid[0].size();
        int res = 0;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(grid[i][j] == 1)
                    res = max(res,dfs(grid,i,j,n,m));
            }
        }
        return res;
    }
};
```



# [827. 最大人工岛](https://leetcode-cn.com/problems/making-a-large-island/)