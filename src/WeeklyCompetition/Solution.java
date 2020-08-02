package WeeklyCompetition;

/**
 * @Author ZJ
 * @Date 2020-08-02
 */
public class Solution {
    /*
        第200周周赛，题目不难，完成两道题，还是菜，第四题卡在取模上
        第三题没仔细看题，希望下次能做出三道题。
     */
    //暴力
    public int countGoodTriplets(int[] arr, int a, int b, int c) {
        int len=arr.length;
        int cnt=0;
        for (int k = len-1; k>=2; k--) {
            for (int j = k-1; j >=1; j--) {
                for (int i = j-1; i >=0 ; i--) {
                    if(Math.abs(arr[i]-arr[j])<=a&&Math.abs(arr[j]-arr[k])<=b&&Math.abs(arr[i]-arr[k])<=c){
                        cnt++;
                    }
                }
            }
        }
        return cnt;
    }

    //模拟计算即可
    public int getWinner(int[] arr, int k) {
        int cnt=0;
        int i=1;
        while(cnt!=k){
            if(arr[i]<arr[0]) {
                cnt++;
                i++;
            }
            else{
                cnt=1;
                int t=arr[0];
                arr[0]=arr[i];
                arr[i]=t;
                i++;
            }
            if(i==arr.length)i=1;
        }
        return arr[0];
    }

    //贪心 预处理统计每行从右往左连续出现0的次数 对于每一行i 如果从i到n-1行没有满足cntZero[j]>=n-i-1返回-1
    //如果存在，就依次交换cntZero[j]到cntZero[i],统计交换次数。
    public int minSwaps(int[][] grid) {
        int n=grid.length;
        int cntZero[] = new int[n];
        for (int i = 0; i < n; i++) {
            int  cnt=0;
            for (int j = n-1; j >0 ; j--) {
                if(grid[i][j]==0)cnt++;
                else break;
            }
            cntZero[i]=cnt;
        }
        int count=0;
        for (int i = 0; i < n; i++) {
            if(cntZero[i]<n-i-1){
                int j=i;
                while(j<n){
                    if(cntZero[j]>=n-i-1)break;
                    j++;
                }
                if(j==n)return -1;//找不到满足条件的，返回-1;
                while(j>i){
                    int temp =cntZero[j];
                    cntZero[j]=cntZero[j-1];
                    cntZero[j-1]=temp;
                    j--;
                    count++;
                }
            }
        }
        return count;
    }

    //双指针 没遇到相同元素比较前不能取模
    public int maxSum(int[] nums1, int[] nums2) {
        long sum1=0,sum2=0,curmaxsum=0;
        int MOD=1000000000+7;
        if(nums1[0]>nums2[0]){
            int []t=nums2;
            nums2=nums1;
            nums1=t;
        }
        int i=0,j=0;
        while(i<nums1.length&&j<nums2.length){
            if(nums1[i]<nums2[j]){
                sum1=sum1+nums1[i++];//这里不能取模还没比较大小
            }else if(nums1[i]==nums2[j]){
                curmaxsum=(Math.max(sum1,sum2)+nums1[i++])%MOD;
                sum1=curmaxsum;
                sum2=curmaxsum;
                j++;
            }
            else{
                sum2=sum2+nums2[j++];////这里不能取模还没比较大小
            }
        }
        if(j<nums2.length){
            while(j<nums2.length)sum2=sum2+nums2[j++];//这里不能取模还没比较大小
        }
        if(i<nums1.length){
            while (i<nums1.length)sum1=sum1+nums1[i++];;//这里不能取模还没比较大小
        }
        curmaxsum= Math.max(sum1,sum2)%MOD;
        return (int)curmaxsum;
    }
}
