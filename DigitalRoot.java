import java.util.Scanner;

public class DigitalRoot {
    
    public static int sumOfDigits(int num) {
        if (num < 10) {
            return num;
        }
        
        int total = 0;
        int remaining = num;
        while (remaining > 0) {
            total += remaining % 10;
            remaining /= 10;
        }
        
        if (total > 9) {
            return sumOfDigits(total);
        }
        
        return total;
    }
    
    public static int sumOfDigitsAlt(int num) {
        if (num < 10) {
            return num;
        }
        
        int digitSum = (num % 10) + sumOfDigitsAlt(num / 10);
        
        if (digitSum > 9) {
            return sumOfDigitsAlt(digitSum);
        }
        
        return digitSum;
    }
    
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        
        System.out.print("Enter a number: ");
        int userNumber = input.nextInt();
        
        int result = sumOfDigits(userNumber);
        System.out.println("The digital root is: " + result);
        
        input.close();
    }
}
