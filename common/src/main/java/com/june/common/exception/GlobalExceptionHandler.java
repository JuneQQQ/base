package com.june.common.exception;


import com.june.common.entity.R;
import lombok.extern.slf4j.Slf4j;
import org.springframework.validation.BindException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;

/**
 * @author june
 */
@ControllerAdvice
@Slf4j
@ResponseBody
public class GlobalExceptionHandler {

    @ExceptionHandler(value = BusinessException.class)
    public R<Void> conditionExceptionHandler(BusinessException e) {
        if (log.isDebugEnabled()) {
            log.debug("BusinessException:::" + e.getMessage());
        }
        return R.fail(e.getErrorCodeEnum());
    }

    /**
     * 自定义验证异常(参数传值)
     */
    @ExceptionHandler(BindException.class)
    public R<String> validatedBindException(BindException e) {
        String message = e.getAllErrors().get(0).getDefaultMessage();
        if (log.isDebugEnabled()) {
            log.debug("BindException:::" + e.getMessage());
        }
        return R.fail(message);
    }

    /**
     * 自定义验证异常(json传值)
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public R<String> methodArgumentNotValidException(MethodArgumentNotValidException e) {
        String message = e.getBindingResult().getAllErrors().get(0).getDefaultMessage();
        if (log.isDebugEnabled()) {
            log.debug("MethodArgumentNotValidException:::" + e.getMessage());
        }
        return R.fail(message);
    }


    @ExceptionHandler(value = Exception.class)
    public R<Void> unknowException(Exception e) {
        log.error("捕捉到未知异常：" + e.getMessage());
        return R.fail(ErrorCodeEnum.UNKNOW_ERROR);
    }
}
