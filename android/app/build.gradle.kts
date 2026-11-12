plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val ortVersion = providers.gradleProperty("wakeitup.onnxruntimeVersion").get()

android {
    namespace = "dev.wakeitup"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.wakeitup"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            // Phones (arm64) + PC emulator (x86_64). Drops ~half the ONNX Runtime native size.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            // ONNX Runtime needs keep rules before minification can be enabled.
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        buildConfig = true
    }

    androidResources {
        // Memory-map friendly and faster to load.
        noCompress += "onnx"
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:$ortVersion")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
}
