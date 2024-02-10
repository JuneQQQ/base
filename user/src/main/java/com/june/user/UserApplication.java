package com.june.user;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.core.env.Environment;

@SpringBootApplication
@ComponentScan("com.june")
public class UserApplication {

    private static final Logger LOG = LoggerFactory.getLogger(UserApplication.class);

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(UserApplication.class);
        Environment env = app.run(args).getEnvironment();
        LOG.info("测试地址: \thttp://127.0.0.1:{}{}/test", env.getProperty("server.port"), env.getProperty("server.servlet.context-path"));
    }

}
