package com.june.user.controller;

import com.june.common.entity.R;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author june
 */
@RestController
public class TestController {
    @RequestMapping("/test")
    public R<Void> test(){
        return R.ok();
    }
}
