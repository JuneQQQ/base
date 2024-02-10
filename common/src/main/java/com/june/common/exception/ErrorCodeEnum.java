package com.june.common.exception;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * @author june
 */

@Getter
@AllArgsConstructor
public enum ErrorCodeEnum {

    /**
     * 正确执行后的返回
     */
    OK(200, ""),

    /**
     * 一级宏观错误码，用户端错误
     */
    USER_ERROR(1000, "用户端错误"),
    REGISTER_AGAIN_ERROR(1001, "重复注册"),
    REQUEST_CODE_TOO_FREQUENT(1002, "60s内重复请求短信接口"),
    LOGIN_CODE_ERROR(1003, "验证码错误或已过期"),

    TOKEN_OR_SECRET_KEY_ERROR(1004, "Token错误"),
    TOKEN_HAS_EXPIRED(1005, "Token已过期"),
    NOT_LOGIN(1006, "未登录"),


    LOGIN_FORM_PARAM_ERROR(1007, "登录传参错误"),
    PHONE_OR_PASSWORD_ERROR(1008, "用户名或密码错误"),

    /**
     * 一级宏观错误码，服务端错误
     */
    SERVER_ERROR(3000, "服务端错误"),

    // 攻击性错误
    ATTACK_ERROR(8888, "攻击性错误"),
    // 未知错误
    UNKNOW_ERROR(9999, "未知错误"),
    ;


    /**
     * 错误码
     */
    final Integer code;

    /**
     * 中文描述
     */
    final String message;
}
