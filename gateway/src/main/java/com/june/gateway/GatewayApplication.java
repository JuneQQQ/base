package com.june.gateway;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.core.env.Environment;

@SpringBootApplication
public class GatewayApplication {

//    private static final Logger LOG = LoggerFactory.getLogger(GatewayApplication.class);

    public static void main(String[] args) {
        new SpringApplication(GatewayApplication.class).run(args);
//        LOG.info("测试地址: \thttp://127.0.0.1:{}{}/test", env.getProperty("server.port"), env.getProperty("server.servlet.context-path"));
    }
}
