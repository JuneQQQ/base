package com.june.common.constant;
/**
 * @author june
 */
public class UserConstant {
    public static final String REDIS_USER_SYS_PREFIX = "user:";
    public static final String REDIS_VERIFICATION_CODE = REDIS_USER_SYS_PREFIX + "code:";
    public static final String REDIS_TOKEN_BLACKLIST = REDIS_USER_SYS_PREFIX + "token-blacklist:";
}
