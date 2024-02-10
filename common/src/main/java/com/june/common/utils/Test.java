package com.june.common.utils;

import cn.hutool.core.date.LocalDateTimeUtil;

import java.time.LocalDateTime;
import java.time.ZoneId;

public class Test {
    public static void main(String[] args) {
        LocalDateTime d = LocalDateTimeUtil.now();
        System.out.println(d);
    }
}
