package com.june.common.exception;

import com.june.common.exception.ErrorCodeEnum;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * @author june
 * 规定：此异常类的Message可直接返回给前台
 * 若直接定义RuntimeException，则不可返回
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class BusinessException extends RuntimeException {

    final ErrorCodeEnum errorCodeEnum;

    public BusinessException(ErrorCodeEnum errorCodeEnum) {
        // 不调用父类 Throwable的fillInStackTrace() 方法生成栈追踪信息，提高应用性能
        // 构造器之间的调用必须在第一行
        super(errorCodeEnum.getMessage(), null, false, false);
        this.errorCodeEnum = errorCodeEnum;
    }

}
