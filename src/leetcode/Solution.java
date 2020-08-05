package leetcode;

import java.util.*;

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

    //104. 二叉树的最大深度 递归
    public int maxDepth(TreeNode root) {
        return root==null?0:Math.max(maxDepth(root.left),maxDepth(root.right))+1;
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

    //114. 二叉树展开为链表
    public void flatten(TreeNode root) {
        TreeNode curNode = root;
        while(curNode!=null){
            if(curNode.left==null){
                curNode=curNode.right;
            }
            else{
                TreeNode preNode=curNode.left;
                while(preNode.right!=null){
                    preNode=preNode.right;
                }
                preNode.right=curNode.right;
                curNode.right=curNode.left;
                curNode.left=null;
            }
        }
    }

    //124. 二叉树中的最大路径和
    /* 最大路径和可能来自三种情况：
     * 1、来自左树的最大路径和
     * 2、来自右树的最大路径和
     * 3、左树到当前节点的最大路径+右树到当前节点的最大路+当前节点的值    注意有负数的情况
     */
    public int maxPathSum(TreeNode root) {
        return getMaxPathSum(root).MaxPathSum;
    }
    public ReturnInfo getMaxPathSum(TreeNode node){
        if(node==null){
            return new ReturnInfo(0, Integer.MIN_VALUE);
        }
         ReturnInfo left = getMaxPathSum(node.left);
         ReturnInfo right = getMaxPathSum(node.right);

         int curHeightPath=left.curHeightMaxPath>right.curHeightMaxPath?left.curHeightMaxPath:right.curHeightMaxPath;
         return new ReturnInfo((curHeightPath>0?curHeightPath:0)+node.val,
                 Math.max(Math.max(left.MaxPathSum,right.MaxPathSum),(left.curHeightMaxPath>0?left.curHeightMaxPath:0)+(right.curHeightMaxPath>0?right.curHeightMaxPath:0)+node.val));
    }
    public static class ReturnInfo{
        int curHeightMaxPath;
        int MaxPathSum;
        public ReturnInfo(int curHeightMaxPath, int MaxPathSum) {
            this.curHeightMaxPath = curHeightMaxPath;
            this.MaxPathSum = MaxPathSum;
        }
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

    //326. 3的幂  换底公式
    public boolean isPowerOfThree(int n) {
        double t=Math.log10(n)/Math.log10(3);
        return t%1==0;
    }
    /*public boolean isPowerOfThree(int n) {
        if(n==0)return false;
        while(n%3==0){
            n/=3;
        }
        return n==1;
    }*/

    //329. 矩阵中的最长递增路径
    /*
    DFS
     */
    public int longestIncreasingPath(int[][] matrix) {
        int row =matrix.length;
        if(row==0)return 0;
        int column = matrix[0].length;
        if(column==0)return 0;
        int res=1;
        int help[][] = new int[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                res = Math.max(res ,dfs(matrix,help,i,j));
            }
        }
        return res;
    }
    public int dfs(int matrix[][], int help[][],int r,int c){
        if(help[r][c]>0)return help[r][c];
        int d[]= {0,1,0,-1,0};//四个方向增量 如(0,1),(-1,0)
        int row =matrix.length;
        int column = matrix[0].length;
        int ans=1;
        for (int i = 0; i < 4; i++) {
            int nextr=r+d[i],nextc=c+d[i+1];//下一个点 四个方向增量 如(0,1),(-1,0)
            if(nextr<0||nextc<0||nextr>=row||nextc>=column)continue;
            if(matrix[r][c]>=matrix[nextr][nextc])continue;
            ans=Math.max(ans,1+dfs(matrix,help,nextr,nextc));
        }
        help[r][c]=ans;
        return ans;
    }

    //337. 打家劫舍 III  暴力递归
    HashMap<TreeNode,Integer> robmap = new HashMap<>();
    HashMap<TreeNode ,Integer> norobmap = new HashMap<>();
    public int rob(TreeNode root) {

        return Math.max(robprocess(root,true),robprocess(root,false));
    }
    public int robprocess(TreeNode node ,boolean flag ){//flag标记当前节点是被偷盗
        if(node == null) return 0;
        if(flag){//被偷盗，则左右子节点不能被盗
            if(!robmap.containsKey(node)){//判断当前节点被盗的最大值。
                robmap.put(node,node.val+robprocess(node.left,false)+robprocess(node.right,false));
            }
            return robmap.get(node);
        }
        else{//没被偷盗，从左右子节点中，选出被盗或没被盗的情况相加
            if(!norobmap.containsKey(node))
                norobmap.put(node,Math.max(robprocess(node.right,true),robprocess(node.right,false))+Math.max(robprocess(node.left,true),robprocess(node.left,false)));
            return norobmap.get(node);
        }
    }

    //392. 判断子序列  双指针
    public boolean isSubsequence(String s, String t) {
        int slen=s.length();
        int tlen=t.length();
        int j=0;
        for (int i = 0; i < tlen; i++) {
            if(slen>j&&t.charAt(i)==s.charAt(j))j++;
        }
        return j==slen;
    }

    //410. 分割数组的最大值 二分查找  最小的最大值一定在maxnum 和 sum之间 通过二分，找到使得子数组和最小的最大值
    public int splitArray(int[] nums, int m) {
        int r=0;
        int l=nums[0];
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            r+=nums[i];
            if(nums[i]>l)l=nums[i];
        }
        while(l<r){
            int mid =l+(r-l)/2;
            int cnt=0;
            int cursum=0;
            for (int i = 0; i < len; i++) {
                if(cursum+nums[i]>mid) {
                    cursum = nums[i];
                    cnt++;
                }
                else{
                    cursum+=nums[i];
                }
            }
            cnt++;
            if(cnt>m)l=mid+1;
            else r=mid;
        }
        return l;
    }

    //415 字符串相加
    public String addStrings(String num1, String num2) {
        StringBuffer res = new StringBuffer();
        int i=num1.length()-1;
        int j=num2.length()-1;
        int car=0;
        int temp;
        while(i>=0||j>=0||car!=0){
            int n1=i>=0?num1.charAt(i--)-'0':0;
            int n2=j>=0?num2.charAt(j--)-'0':0;
            temp=n1+n2+car;
            car=temp/10;
            res.append(temp%10);
        }
        return res.reverse().toString();
    }

    //632. 最小区间 从 k 个列表中各取一个数，使得这 k 个数中的最大值与最小值的差最小。
    // 优先队列、堆  小顶堆维护当前最小值指针索引  同时curMax找到当前最大值
    public int[] smallestRange(List<List<Integer>> nums) {
        int k = nums.size();
        int p[]=new int[k];
        int maxNum,minRange,minNum;
        PriorityQueue<Integer> minQueue = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return nums.get(o1).get(p[o1])-nums.get(o2).get(p[o2]);
            }
        });
        maxNum = nums.get(0).get(0);
        for (int i = 0; i < k; i++) {
            minQueue.offer(i);
            maxNum = Math.max(maxNum,nums.get(i).get(0));
        }
        minNum = nums.get(minQueue.peek()).get(p[minQueue.peek()]);
        minRange = maxNum-minNum;
        int curMaxNum = maxNum;
        while(true){
            int minIndex= minQueue.poll();
            int curRange=curMaxNum-nums.get(minIndex).get(p[minIndex]);
            if(curRange<minRange){//curRange小才更新最小值区间
                minRange =curRange;
                minNum = nums.get(minIndex).get(p[minIndex]);
                maxNum = curMaxNum;
            }
            p[minIndex]++;//当前最小值指针后移
            if(p[minIndex]==nums.get(minIndex).size()) break;//当某一列表到末尾时退出
            minQueue.offer(minIndex);//更新当前最小最小值指针
            curMaxNum = Math.max(curMaxNum,nums.get(minIndex).get(p[minIndex]));
        }
        return new int[]{minNum,maxNum};
    }

    //988. 从叶结点开始的最小字符串
    String smallestFromLeafRes="";
    public String smallestFromLeaf(TreeNode root) {
        smallestFromLeafDFS(new StringBuffer(),root);
        return smallestFromLeafRes;
    }
    public void smallestFromLeafDFS(StringBuffer str, TreeNode node){
        if(node==null)return ;
        str.append((char)('a'+node.val));
        if(node.left==null&&node.right==null){
            String s=str.reverse().toString();
            if(s.compareTo(smallestFromLeafRes)<0||smallestFromLeafRes=="") smallestFromLeafRes=s;
            str.reverse();
        }
        smallestFromLeafDFS(str,node.left);
        smallestFromLeafDFS(str,node.right);
        str.deleteCharAt(str.length()-1);
    }

    //989. 数组形式的整数加法
    public List<Integer> addToArrayForm(int[] A, int K) {
        LinkedList<Integer> res = new LinkedList<>();
        int car = 0;
        int i=A.length-1;
        while(i>=0||car!=0||K!=0) {
            int temp = i<0?car+K%10:A[i--]+car+K%10;
            car = temp/10;
            res.addFirst(temp%10);
            K/=10;
        }
        return res;
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
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}
