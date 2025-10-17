import java.util.Scanner;

public class RecursiveWordReversal {
    
    public static String reverseWithTwist(String word) {
        if (word.isEmpty()) {
            return "";
        }
        
        char currentChar = word.charAt(0);
        
        if (isVowel(currentChar)) {
            currentChar = '*';
        }
        
        return reverseWithTwist(word.substring(1)) + currentChar;
    }
    
    public static boolean isVowel(char letter) {
        letter = Character.toLowerCase(letter);
        return letter == 'a' || letter == 'e' || letter == 'i' || 
               letter == 'o' || letter == 'u';
    }
    
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        
        System.out.print("Enter a word: ");
        String userWord = input.nextLine();
        
        String reversed = reverseWithTwist(userWord);
        System.out.println("Reversed with twist: " + reversed);
        
        input.close();
    }
}
