import java.security.MessageDigest;
import java.io.*;
import java.util.Random;

public class LegacyUtils {

    public static String hashPassword(String password) throws Exception {
        MessageDigest md = MessageDigest.getInstance("MD5");
        byte[] bytes = md.digest(password.getBytes());
        return bytesToHex(bytes);
    }

    public static void runCommand(String userInput) throws IOException {
        Runtime.getRuntime().exec("cmd.exe /c " + userInput);
    }

    public static void readFile(String fileName) throws Exception {
        File file = new File("/data/" + fileName);
        FileInputStream fis = new FileInputStream(file);
    }

    public static int generateToken() {
        Random r = new Random();
        return r.nextInt();
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) sb.append(String.format("%02x", b));
        return sb.toString();
    }
}