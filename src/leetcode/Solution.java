package leetcode;

public class Solution {
    //9. 回文数
    public boolean isPalindrome(int x) {
        if(x<0)return false;
        int cur =0;
        int t=x;
        while(t>0){
            cur=cur*10+t%10;
            t/=10;
        }
        return cur==x;
    }
    //64. 最小路径和
    //dp[i][j]=Math.min(dp[i-1][j],dp[i][j-1])+grid[i][j];
    public int minPathSum(int[][] grid) {
        int m=grid.length;
        int n=grid[0].length;
        int dp[][] = new int[m][n];
        dp[0][0]=grid[0][0];
        for (int i = 1; i <n ; i++) {
            dp[0][i]=grid[0][i]+dp[0][i-1];
        }
        for (int i = 1; i <m ; i++) {
            dp[i][0]=grid[i][0]+dp[i-1][0];
        }
        for (int i = 1; i <m ; i++) {
            for (int j = 1; j <n ; j++) {
                dp[i][j]=Math.min(dp[i-1][j],dp[i][j-1])+grid[i][j];
            }
        }
        return dp[m-1][n-1];
    }

    //96. 不同的二叉搜索树dp
    /*G(n)表示n个点存在二叉排序树的个数，f(i)为以根为i的二叉排序树的个数
    f(i)=G(i-1)*G(n-i)
    G(n)=f(1)+f(2)+.....+f(n)
    */
    public int numTrees(int n) {
        int dp[] = new int[n+1];
        dp[0]=1;
        dp[1]=1;
        for (int i = 2; i <n+1 ; i++) {
            for (int j = 1; j <=i ; j++) {
                dp[i]+=dp[j-1]*dp[i-j];
            }
        }
        return dp[n];
    }

    //167. 两数之和 II - 输入有序数组 双指针
    public int[] twoSum(int[] numbers, int target) {
        int left=0;
        int right=numbers.length-1;
        while(left<right){
            int cursum=numbers[left]+numbers[right];
            if(cursum<target) left++;
            else if(cursum>target) right--;
            else return new int[]{left+1,right+1};
        }
        return new int[2];
    }

    //1025. 除数博弈
    /*
    最终结果应该是占到 2 的赢，占到 1 的输
    若当前为奇数，奇数的约数只能是奇数或者 1，因此下一个一定是偶数
    若当前为偶数， 偶数的约数可以是奇数可以是偶数也可以是 1，因此直接减 1，则下一个是奇数
    因此，奇则输，偶则赢
     */
    public boolean divisorGame(int N) {
        return N%2==0;
    }
   /* dp
     public boolean divisorGame(int N) {
        if(N<=1)return false;
        boolean dp[]=new boolean[N+1];
        dp[1]=false;
        dp[2]=true;
        for (int i = 3; i <=N ; i++) {
            for (int j = 1; j <i/2+1 ; j++) {
                if(i%j==0&&!dp[i-j]){
                    dp[i]=true;
                    break;
                }
            }
        }
        return dp[N];
    }*/



}
