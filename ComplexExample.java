public class ComplexExample {

    public String evaluateOrder(int status, boolean isVip, boolean hasCoupon,
                                 int amount, boolean isWeekend, String region) {
        String result = "";

        if (status == 1) {
            if (isVip) {
                if (hasCoupon) {
                    result = "VIP+쿠폰 할인";
                } else if (amount > 100000) {
                    result = "VIP 고액 할인";
                } else {
                    result = "VIP 기본 할인";
                }
            } else if (hasCoupon) {
                result = "쿠폰 할인";
            } else {
                result = "일반 주문";
            }
        } else if (status == 2) {
            for (int i = 0; i < amount; i++) {
                if (i % 2 == 0 && isWeekend) {
                    result += "주말 짝수 처리 ";
                } else if (i % 3 == 0 || region.equals("SEOUL")) {
                    result += "서울/3배수 처리 ";
                } else {
                    result += "기본 처리 ";
                }
            }
        } else if (status == 3) {
            while (amount > 0) {
                if (amount % 5 == 0) {
                    amount -= 5;
                } else if (amount % 3 == 0) {
                    amount -= 3;
                } else {
                    amount -= 1;
                }
            }
        } else {
            switch (region) {
                case "SEOUL": result = "서울"; break;
                case "BUSAN": result = "부산"; break;
                case "DAEGU": result = "대구"; break;
                default: result = "기타";
            }
        }

        return result;
    }
}