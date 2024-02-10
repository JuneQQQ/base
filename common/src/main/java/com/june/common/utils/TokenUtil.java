package com.june.common.utils;

import cn.hutool.core.convert.NumberWithFormat;
import cn.hutool.core.date.LocalDateTimeUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.jwt.JWT;
import cn.hutool.jwt.JWTUtil;
import com.june.common.exception.BusinessException;
import com.june.common.exception.ErrorCodeEnum;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.TimeZone;

import static com.june.common.constant.UserConstant.REDIS_TOKEN_BLACKLIST;

/**
 * @author june
 */
public class TokenUtil {
    private static final Long REDIS_TTL_NEGATIVE = -2L;

    private static String getTokenFromCookie(HttpServletRequest request) {
        Cookie[] cookies = request.getCookies();

        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if ("token".equals(cookie.getName())) {
                    return cookie.getValue();
                }
            }
        }
        return null;
    }

    /**
     * @param token               token
     * @param secretKey           密钥
     * @param stringRedisTemplate 此参数是由于需要从redis获取token黑名单
     * @return 成功验证的JWTtoken对象
     */
    public static JWT checkToken(String token, String secretKey, StringRedisTemplate stringRedisTemplate) {
        if (StrUtil.isBlank(token)) {
            throw new BusinessException(ErrorCodeEnum.NOT_LOGIN);
        }

        JWT jwt = JWT.of(token).setKey(secretKey.getBytes());

        if (!jwt.verify()) {
            throw new BusinessException(ErrorCodeEnum.TOKEN_OR_SECRET_KEY_ERROR);
        }

        if (!jwt.validate(0)) {
            throw new BusinessException(ErrorCodeEnum.TOKEN_HAS_EXPIRED);
        }

        if (!REDIS_TTL_NEGATIVE.equals(stringRedisTemplate.getExpire(REDIS_TOKEN_BLACKLIST + token))) {
            throw new BusinessException(ErrorCodeEnum.NOT_LOGIN);
        }

        return jwt;
    }

    /**
     * 此方法依赖机器时钟，若系统时钟不同于MySQL时钟(UTC+8)，会出现时间错误
     */
    private static LocalDateTime convertDate(NumberWithFormat date) {
        return LocalDateTimeUtil.of(date.longValue() * 1000, ZoneId.systemDefault());
    }

    public static String getToken(HttpServletRequest request) {
        String token = StrUtil.blankToDefault(request.getHeader("token"), getTokenFromCookie(request));
        if (StrUtil.isBlank(token)) {
            throw new BusinessException(ErrorCodeEnum.TOKEN_OR_SECRET_KEY_ERROR);
        }
        return token;
    }

    public static LocalDateTime getTokenExpireAt(String token) {
        return convertDate((NumberWithFormat) JWT.of(token).getPayload("exp"));
    }

    public static LocalDateTime getTokenExpireAt(HttpServletRequest request) {
        return convertDate((NumberWithFormat) JWT.of(getToken(request)).getPayload("exp"));
    }
}
