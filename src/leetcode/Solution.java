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

    //


}
