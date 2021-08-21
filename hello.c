#include<stdio.h>
 
int main() {
    int n1[4]={5,2,39,12};
    int temp=0;
    printf("Largest number in the array: \n");
    for(int i = 0; i < 3; i++){
        if(n1[i] < n1[i+1]){
            temp = n1[i+1];
        }
        else{
            temp=n1[i];
        }
    }
    printf("Largest number is %d \n",temp);
   return 0;
}


// temperature of 7 days