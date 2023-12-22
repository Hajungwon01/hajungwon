import java.util.*;

class AES{
    String Plantext = "";
    String Cyphertext = "";
    int Nr = 0;

    public AES(String p){
        this.Plantext = p;
    }

    public void encryption(){
        for (int round = 1; round<Nr; round++)
        {
            SubBytes();
            ShiftRows();
            MixColumns();
            AddRoundKey(round);
        }

            SubBytes();
            ShiftRows();
            AddRoundKey(Nr);
    }

    public void SubBytes(){

    }
    
    public void ShiftRows(){

    }

    public void MixColumns(){

    }

    public void AddRoundKey(int r){

    }

    public static void main(String[] args) {
        AES aes = new AES("hello");

    }
}