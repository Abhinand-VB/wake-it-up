// Pure-Kotlin core: tokenizer, encoder, classifier wrapper, decision logic.
// No Android dependencies, so everything here is unit-tested on the PC (JVM),
// including model parity via the desktop ONNX Runtime build.
plugins {
    id("org.jetbrains.kotlin.jvm")
}

kotlin {
    jvmToolchain(17)
}

val ortVersion = providers.gradleProperty("wakeitup.onnxruntimeVersion").get()

dependencies {
    // Same API as onnxruntime-android; the app module supplies the Android runtime.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    // org.json is built into Android; the JVM needs the standalone artifact.
    compileOnly("org.json:json:20240303")

    testImplementation(kotlin("test-junit"))
    testImplementation("junit:junit:4.13.2")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation("org.json:json:20240303")
}

tasks.test {
    // Model bundle and fixtures live in the app module (see SOURCE_OF_TRUTH.md §7, §12).
    val modelDir = rootProject.file("app/src/main/assets/model")
    val fixturesDir = rootProject.file("app/src/test/resources/fixtures")
    systemProperty("wakeitup.modelDir", modelDir.absolutePath)
    systemProperty("wakeitup.fixturesDir", fixturesDir.absolutePath)
    // Re-run tests when a new model bundle or new fixtures are dropped in.
    inputs.files(fileTree(modelDir), fileTree(fixturesDir)).withPropertyName("modelAndFixtures")
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
