package com.june.common.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import com.june.common.exception.ErrorCodeEnum;
import lombok.Getter;

@Getter
public class R<T> {
    public static void main(String[] args) {
    }

    /**
     * 响应码
     */
    @Schema(description = "错误码，200:没有错误")
    private final Integer code;

    /**
     * 响应消息
     */
    @Schema(description = "响应消息")
    private final String msg;

    /**
     * 响应数据
     */
    @Schema(description = "响应数据")
    private T data;

    private R() {
        this.code = ErrorCodeEnum.OK.getCode();
        this.msg = ErrorCodeEnum.OK.getMessage();
    }

    private R(ErrorCodeEnum errorCode) {
        this.code = errorCode.getCode();
        this.msg = errorCode.getMessage();
    }

    private R(T data) {
        this();
        this.data = data;
    }

    private R(T data, Integer code) {
        this.data = data;
        this.code = code;
        this.msg = "";
    }

    /**
     * 业务处理成功,无数据返回
     */
    public static R<Void> ok() {
        return new R<>();
    }

    /**
     * 业务处理成功，有数据返回
     */
    public static <T> R<T> ok(T data) {
        return new R<>(data);
    }

    /**
     * 业务处理失败
     */
    public static R<Void> fail(ErrorCodeEnum errorCode) {
        return new R<>(errorCode);
    }

    public static R<String> fail(String msg) {
        return new R<>(msg, ErrorCodeEnum.UNKNOW_ERROR.getCode());
    }

    /**
     * 系统错误
     */
    public static R<Void> error() {
        return new R<>(ErrorCodeEnum.SERVER_ERROR);
    }

}
