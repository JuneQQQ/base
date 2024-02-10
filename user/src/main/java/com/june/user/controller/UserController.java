package com.june.user.controller;

import com.june.common.entity.R;
import com.june.user.pojo.dao.entity.User;
import com.june.user.pojo.vo.LoginFormVo;
import com.june.user.service.IUserService;
import jakarta.annotation.Resource;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@RestController
public class UserController {
    @Resource
    public IUserService userService;

    @GetMapping("send-code")
    public R<String> sendCode(@RequestParam String phone){
        return R.ok(userService.sendCode(phone));
    }


    @PostMapping("login")
    public R<String> login(@Validated @RequestBody LoginFormVo vo){
        return R.ok(userService.login(vo));
    }

    @PostMapping("info")
    public R<User> getUserInfo(HttpServletRequest request){
        return R.ok(userService.getUserInfo(request));
    }

    @PostMapping("logout")
    public R<Void> logout(HttpServletRequest request){
        userService.logout(request);
        return R.ok();
    }
}
