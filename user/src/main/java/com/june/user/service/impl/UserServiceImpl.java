package com.june.user.service.impl;

import cn.hutool.core.convert.NumberWithFormat;
import cn.hutool.core.date.DateField;
import cn.hutool.core.date.DateUtil;
import cn.hutool.core.date.LocalDateTimeUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.core.util.RandomUtil;
import cn.hutool.jwt.JWT;
import cn.hutool.jwt.JWTPayload;
import cn.hutool.jwt.JWTUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.june.common.exception.BusinessException;
import com.june.common.exception.ErrorCodeEnum;
import com.june.user.pojo.dao.entity.User;
import com.june.user.pojo.dao.mapper.UserMapper;
import com.june.user.pojo.vo.LoginFormVo;
import com.june.user.service.IUserService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import jakarta.annotation.Resource;
import jakarta.servlet.http.HttpServletRequest;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Date;

import static com.june.common.constant.UserConstant.REDIS_TOKEN_BLACKLIST;
import static com.june.common.constant.UserConstant.REDIS_VERIFICATION_CODE;
import static com.june.common.utils.TokenUtil.*;

/**
 * <p>
 * 会员 服务实现类
 * </p>
 *
 * @author June
 */
@Service
public class UserServiceImpl extends ServiceImpl<UserMapper, User> implements IUserService {
    @Resource
    StringRedisTemplate stringRedisTemplate;


    @Value("${jwt.secret-key}")
    @SuppressWarnings("all")
    String JWTSecretKey;
    @Resource
    UserMapper userMapper;

    @Override
    public String sendCode(String phone) {
//        String randomNumber = RandomUtil.randomNumbers(4);
        String code = "6666";
        stringRedisTemplate.opsForValue().setIfAbsent(REDIS_VERIFICATION_CODE + phone, code, Duration.ofSeconds(60));
        return code;
    }


    @Override
    @Transactional(rollbackFor = Throwable.class)
    public String login(LoginFormVo vo) {
        return switch (vo.loginBy()) {
            case "password" -> loginByPassword(vo);
            case "code" -> loginByCode(vo);
            default -> throw new BusinessException(ErrorCodeEnum.LOGIN_FORM_PARAM_ERROR);
        };
    }

    /**
     * 通过手机号登录，未登录会进行注册
     *
     * @return 脱敏过的User对象
     */
    private String loginByCode(LoginFormVo vo) {
        String code = stringRedisTemplate.opsForValue().get(REDIS_VERIFICATION_CODE + vo.phone());

        if (log.isDebugEnabled()) {
            log.debug(vo.toString());
        }

        User userByDb = userMapper.selectOne(new LambdaQueryWrapper<User>().eq(User::getPhone, vo.phone()));

        // 不存在，注册
        if (ObjectUtil.isNull(userByDb)) {
            userByDb = new User();
            BeanUtils.copyProperties(vo, userByDb);
            // 初始化 nickname
            userByDb.setNickname("乘客" + RandomUtil.randomNumbers(4));
            register(userByDb);
        }

        if (StringUtils.equals(vo.code(), code)) {
            return generateToken(userByDb);
        }
        throw new BusinessException(ErrorCodeEnum.LOGIN_CODE_ERROR);
    }

    private String loginByPassword(LoginFormVo vo) {
        // TODO 数据库密码要加密存储
        User userByDb = userMapper.selectOne(new LambdaQueryWrapper<User>()
                .eq(User::getPhone, vo.phone())
                .eq(User::getPassword, vo.password())
        );
        if (ObjectUtil.isNotNull(userByDb)) {
            return generateToken(userByDb);
        }
        throw new BusinessException(ErrorCodeEnum.PHONE_OR_PASSWORD_ERROR);
    }

    private String generateToken(User user) {
        // TODO 改进双token::需要前端支持
//            JWTPayload longPayload = new JWTPayload()
//                    .setIssuer("June")
//                    .setPayload("phone",vo.phone())
//                    .setExpiresAt(DateUtil.offset(new Date(), DateField.MINUTE,10));
//            String longToken = JWTUtil.createToken(shortPayload.getClaimsJson(), JWTSecretKey.getBytes());
        JWTPayload shortPayload = new JWTPayload()
                .setIssuer("Admin")
                .setPayload("id", user.getId())
                .setPayload("phone", user.getPhone())
                .setPayload("nickname", user.getNickname())
                .setPayload("avatar", user.getAvatar())
                .setPayload("createTime", user.getCreateTime())
                .setExpiresAt(DateUtil.offset(new Date(), DateField.MINUTE, 30));
        return JWTUtil.createToken(shortPayload.getClaimsJson(), JWTSecretKey.getBytes());
    }


    @Override
    public Long register(User user) {
        if (ObjectUtil.isNotNull(baseMapper.selectOne(new LambdaQueryWrapper<User>().eq(User::getPhone, user.getPhone())))) {
            throw new BusinessException(ErrorCodeEnum.REGISTER_AGAIN_ERROR);
        }
        if (baseMapper.insert(user) != 1) {
            throw new RuntimeException("register DB插入出错");
        }
        return user.getId();
    }

    @Override
    public User getUserInfo(HttpServletRequest request) {
        String token = getToken(request);

        JWT jwt = checkToken(token, JWTSecretKey, stringRedisTemplate);

        User user = new User();
        user.setId(((NumberWithFormat) jwt.getPayload("id")).longValue());
        user.setNickname((String) jwt.getPayload("nickname"));
        user.setPhone((String) jwt.getPayload("phone"));
        user.setAvatar((String) jwt.getPayload("avatar"));
        user.setCreateTime(LocalDateTimeUtil.of(1707409471L * 1000, ZoneId.of("Asia/Shanghai")));

        return user;
    }

    @Override
    public void logout(HttpServletRequest request) {
        String token = getToken(request);
        LocalDateTime expireAt = getTokenExpireAt(token);

        var gap = LocalDateTimeUtil.between(LocalDateTime.now(), expireAt);

        if (!gap.isNegative()) {
            stringRedisTemplate.opsForValue().setIfAbsent(REDIS_TOKEN_BLACKLIST + token, getUserInfo(request).getId().toString(), gap);
        }
    }


}
