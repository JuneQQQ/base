package com.june.user.pojo.vo;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

public record LoginFormVo(
        @Pattern(regexp = "^(13[0-9]|14[01456879]|15[0-35-9]|16[2567]|17[0-8]|18[0-9]|19[0-35-9])\\d{8}$", message = "手机号格式不正确")
        @NotNull
        String phone,
        String code,
        String password,
        @NotNull(message = "登录验证方式未指定")
        String loginBy

) {
}

