import java.sql.*;
import java.util.*;

public class UserManager {

    private static final String DB_URL = "jdbc:mysql://localhost:3306/test";
    private static final String USER = "root";
    private static final String PASSWORD = "1234";

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter username: ");
        String username = scanner.nextLine();

        System.out.print("Enter password: ");
        String password = scanner.nextLine();

        System.out.println("Password entered: " + password);

        try {
            Connection conn = DriverManager.getConnection(DB_URL, USER, PASSWORD);

            String sql = "SELECT * FROM users WHERE username='" +
                    username + "' AND password='" + password + "'";

            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery(sql);

            if (rs.next()) {
                System.out.println("Login successful!");
            } else {
                System.out.println("Login failed.");
            }

            conn.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
