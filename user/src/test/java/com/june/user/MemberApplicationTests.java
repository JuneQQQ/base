package com.june.user;

import cn.hutool.core.date.DateField;
import cn.hutool.core.date.DateUtil;
import cn.hutool.jwt.JWT;
import cn.hutool.jwt.JWTPayload;
import cn.hutool.jwt.JWTUtil;
import cn.hutool.jwt.JWTValidator;
import cn.hutool.jwt.signers.JWTSigner;
import cn.hutool.jwt.signers.JWTSignerUtil;
import com.june.user.pojo.dao.entity.User;
import com.june.user.pojo.dao.mapper.UserMapper;
import jakarta.annotation.Resource;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.redis.core.RedisCallback;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.Date;
import java.util.Random;


@SpringBootTest
class UserApplicationTests {
    @Resource
    UserMapper userMapper;
    @Resource
    StringRedisTemplate stringRedisTemplate;
    @Value("${jwt.secret-key}")
    String key;
    @Test
    void contextLoads() {
        User user = userMapper.selectOne(null);
        System.out.println(user);
    }

    @Test
    void testRedis(){
        System.out.println(stringRedisTemplate.getExpire("ABC"));
    }
    public void populateRedisWithRandomData() {
        Random random = new Random();
        stringRedisTemplate.executePipelined((RedisCallback<Object>) connection -> {
            for (int i = 0; i < 1000_0000; i++) {
                String key = "key-" + i;
                String value = "value-" + random.nextInt(100);
                connection.set(key.getBytes(), value.getBytes());
            }
            return null;
        });
    }

    public static void main(String[] args) {
        JWTPayload shortPayload = new JWTPayload()
                .setIssuer("June")
                .setPayload("phone","10000000000")
                .setExpiresAt(DateUtil.offset(new Date(),DateField.SECOND,-5));

        // 可以设置其他自定义的键值对
        String token = JWTUtil.createToken(shortPayload.getClaimsJson(), "abc".getBytes());
//        System.out.println(token);

        JWT jwt = JWT.of(token).setKey("abc".getBytes());


        try {
            jwt.validate(0);
            jwt.verify();

            Long id = (Long) jwt.getPayload("id");
            System.out.println(id);
            String nickname = (String) jwt.getPayload("phone");
            System.out.println(nickname);
//            JWTValidator.of(jwt).validateDate();
        } catch (Exception e) {
            e.printStackTrace();
        }

//        System.out.println(jwt.verify());



//        System.out.println(JWTUtil.verify(token, "abc".getBytes()));
    }
}
