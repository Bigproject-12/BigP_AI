import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class SecureExample {

    // 취약: 사용자 입력을 문자열 그대로 이어붙여서 쿼리 실행
    public ResultSet findUserByName(Connection conn, String userName) throws SQLException {
        Statement stmt = conn.createStatement();
        String sql = "SELECT * FROM users WHERE name = '" + userName + "'";
        return stmt.executeQuery(sql);   // ← guardrail-java-sqli-dynamic-execute 탐지 지점
    }

    // 취약: executeUpdate에도 동일한 패턴
    public int deleteUserById(Connection conn, String userId) throws SQLException {
        Statement stmt = conn.createStatement();
        String sql = "DELETE FROM users WHERE id = " + userId;
        return stmt.executeUpdate(sql);  // ← 탐지 지점
    }

    // 취약: execute()도 동일하게 잡힘
    public boolean runQuery(Connection conn, String tableName) throws SQLException {
        Statement stmt = conn.createStatement();
        return stmt.execute("SELECT * FROM " + tableName);  // ← 탐지 지점
    }

    public ResultSet findUserByVariable(Connection conn, String preparedSql) throws SQLException {
        Statement stmt = conn.createStatement();
        return stmt.executeQuery(preparedSql);  // 변수를 그대로 전달, + 연산자 없음
    }
}