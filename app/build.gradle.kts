plugins {
    id("com.enpasos.bugs.java-conventions")
    id("idea")
    id("java")
}

dependencies {

    implementation(libs.bundles.djl)

    implementation(libs.commons.cli)

    implementation(libs.slf4j.api)
    runtimeOnly(libs.slf4j.simple)

    compileOnly(libs.lombok)
    annotationProcessor(libs.lombok)
    testCompileOnly(libs.lombok)
    testAnnotationProcessor(libs.lombok)

}

description = "app"


tasks.withType<Jar> {
    manifest {
        attributes["Main-Class"] = "com.enpasos.bugs.Main"
    }
    from(configurations.runtimeClasspath.get().map {if (it.isDirectory) it else zipTree(it)})
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}
