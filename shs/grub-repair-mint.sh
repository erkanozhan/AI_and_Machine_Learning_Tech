#!/usr/bin/env bash
# ============================================================================
# grub-repair-mint.sh
# ----------------------------------------------------------------------------
# Linux Mint (Ubuntu tabanlı) UEFI sistemlerinde, başka bir kurulumun GRUB'ı
# eski EFI partition'ına yazdığında çalışan kurtarma scripti.
#
# Senaryo:
#   - Gigabyte makinesinde sda diskinde dual-boot Mint + Windows kurulu.
#   - Bir USB'ye Mint kurulumu yapıldı; installer GRUB'ı yanlışlıkla
#     sda'daki EFI partition'a da yazdı, dolayısıyla USB takılı olmadığında
#     boot GRUB rescue'da takılıyor.
#   - Bu script bir Mint Live ortamından çalıştırılır ve sda5'teki mevcut
#     Mint kurulumunun GRUB'ını yeniden kurar.
#
# Kullanım:
#   1) Mint Live USB'den boot edin (Ventoy + Mint ISO).
#   2) Terminal açın, scripti indirin veya kopyalayın.
#   3) chmod +x grub-repair-mint.sh
#   4) sudo ./grub-repair-mint.sh
#
# Bu script PROD makinede çalıştırılmadan önce script içindeki sabitleri
# (özellikle ROOT_PART, EFI_PART, TARGET_DISK) kendi sisteminize göre
# doğrulayın. Varsayılanlar Erkan'ın Gigabyte sistemine göredir.
# ============================================================================

set -euo pipefail

# ---- Sabitler --------------------------------------------------------------
TARGET_DISK="/dev/sda"             # GRUB'ın yazılacağı disk
ROOT_PART="/dev/sda5"              # Mint kök dosya sistemi
EFI_PART="/dev/sda1"               # EFI System Partition (FAT32)
MNT="/mnt/mint"                    # chroot için mount noktası
EXPECTED_ROOT_UUID="30e9bbf3-d14a-4f83-b140-cd66622a39f2"
EXPECTED_EFI_UUID="0A16-137C"

# ---- Yardımcılar -----------------------------------------------------------
log()  { printf '\n\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\n\033[1;33m[UYARI]\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31m[HATA]\033[0m %s\n' "$*" >&2; exit 1; }

require_root() {
    [[ $EUID -eq 0 ]] || die "Bu script root yetkisi ister: sudo ./$(basename "$0")"
}

confirm() {
    local prompt="$1"
    read -r -p "$prompt [evet/hayir]: " answer
    [[ "$answer" == "evet" ]] || die "Kullanıcı onayı alınamadı, çıkılıyor."
}

# ---- Ön kontroller ---------------------------------------------------------
preflight() {
    log "Ön kontroller yapılıyor..."

    # Live ortam mı?
    if ! findmnt /rofs >/dev/null 2>&1; then
        warn "Bu script bir Live ortamdan çalıştırılmak üzere tasarlandı."
        warn "/rofs mount noktası bulunamadı, beklenmedik bir ortamda olabilirsiniz."
        confirm "Yine de devam edilsin mi?"
    fi

    # Hedef bölümler var mı?
    [[ -b "$ROOT_PART" ]] || die "$ROOT_PART bulunamadı. Diskler değişmiş olabilir; lsblk -f ile kontrol edin."
    [[ -b "$EFI_PART"  ]] || die "$EFI_PART bulunamadı. Diskler değişmiş olabilir; lsblk -f ile kontrol edin."
    [[ -b "$TARGET_DISK" ]] || die "$TARGET_DISK bulunamadı."

    # UUID doğrulaması — yanlış diske GRUB yazmamak için kritik
    local actual_root_uuid actual_efi_uuid
    actual_root_uuid="$(blkid -s UUID -o value "$ROOT_PART" || true)"
    actual_efi_uuid="$(blkid -s UUID -o value "$EFI_PART" || true)"

    if [[ "$actual_root_uuid" != "$EXPECTED_ROOT_UUID" ]]; then
        warn "Beklenen kök UUID: $EXPECTED_ROOT_UUID"
        warn "Gerçek kök UUID:    $actual_root_uuid"
        warn "Bu büyük olasılıkla yanlış disk veya farklı bir sistem demek."
        confirm "Yine de bu bölüme GRUB kurulsun mu?"
    else
        log "Kök UUID doğrulandı: $actual_root_uuid"
    fi

    if [[ "$actual_efi_uuid" != "$EXPECTED_EFI_UUID" ]]; then
        warn "Beklenen EFI UUID: $EXPECTED_EFI_UUID"
        warn "Gerçek EFI UUID:    $actual_efi_uuid"
        confirm "Yine de devam edilsin mi?"
    else
        log "EFI UUID doğrulandı: $actual_efi_uuid"
    fi

    # ROOT_PART halihazırda mount mu?
    if findmnt "$ROOT_PART" >/dev/null 2>&1; then
        warn "$ROOT_PART zaten mount edilmiş durumda."
        findmnt "$ROOT_PART"
        confirm "Mevcut mount'u kullanmaya devam edilsin mi?"
    fi
}

# ---- Mount işlemleri -------------------------------------------------------
mount_filesystems() {
    log "Dosya sistemleri mount ediliyor..."

    mkdir -p "$MNT"

    if ! findmnt "$MNT" >/dev/null 2>&1; then
        mount "$ROOT_PART" "$MNT"
    fi

    # Kök dizin içeriği makul mü?
    if [[ ! -d "$MNT/boot" || ! -d "$MNT/etc" || ! -d "$MNT/usr" ]]; then
        die "$MNT içinde beklenen Linux kök yapısı yok. Yanlış bölüm mount edilmiş olabilir."
    fi

    # EFI partition
    mkdir -p "$MNT/boot/efi"
    if ! findmnt "$MNT/boot/efi" >/dev/null 2>&1; then
        mount "$EFI_PART" "$MNT/boot/efi"
    fi

    # Sistem cihazlarını bind mount et (grub-install donanıma erişebilsin)
    for d in dev dev/pts proc sys run; do
        mkdir -p "$MNT/$d"
        if ! findmnt "$MNT/$d" >/dev/null 2>&1; then
            mount --bind "/$d" "$MNT/$d"
        fi
    done

    # UEFI runtime variables (efibootmgr için gerekli)
    if [[ -d /sys/firmware/efi/efivars ]]; then
        if ! findmnt "$MNT/sys/firmware/efi/efivars" >/dev/null 2>&1; then
            mount --bind /sys/firmware/efi/efivars "$MNT/sys/firmware/efi/efivars" 2>/dev/null || \
                warn "efivars bind mount başarısız, devam ediliyor."
        fi
    else
        warn "/sys/firmware/efi/efivars yok — sistem Legacy modda boot etmiş olabilir."
        warn "GRUB'ın UEFI tarafına temiz yazılabilmesi için Live ortamı UEFI modda boot etmiş olmalısınız."
        confirm "Yine de devam edilsin mi?"
    fi

    log "Mount işlemleri tamamlandı."
    findmnt -R "$MNT" || true
}

# ---- Cleanup (her durumda çalışır) -----------------------------------------
cleanup() {
    log "Temizlik: mount'lar sökülüyor..."

    # Ters sırayla unmount
    for target in \
        "$MNT/sys/firmware/efi/efivars" \
        "$MNT/run" \
        "$MNT/sys" \
        "$MNT/proc" \
        "$MNT/dev/pts" \
        "$MNT/dev" \
        "$MNT/boot/efi" \
        "$MNT"
    do
        if findmnt "$target" >/dev/null 2>&1; then
            umount -R "$target" 2>/dev/null || umount -l "$target" 2>/dev/null || true
        fi
    done

    log "Temizlik tamamlandı."
}
trap cleanup EXIT

# ---- GRUB onarımı ----------------------------------------------------------
repair_grub() {
    log "chroot içinde grub-install ve update-grub çalıştırılıyor..."

    chroot "$MNT" /bin/bash -e <<EOF
set -e
echo
echo "===> grub-install $TARGET_DISK"
grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=ubuntu --recheck $TARGET_DISK

echo
echo "===> update-grub"
update-grub

echo
echo "===> efibootmgr -v"
efibootmgr -v || true
EOF

    log "GRUB onarımı tamamlandı."
}

# ---- Ana akış --------------------------------------------------------------
main() {
    require_root

    cat <<EOF

============================================================================
  Linux Mint GRUB Kurtarma Scripti
  Hedef disk:    $TARGET_DISK
  Kök bölümü:    $ROOT_PART
  EFI bölümü:    $EFI_PART
  Mount noktası: $MNT
============================================================================

Bu script $TARGET_DISK üzerine GRUB bootloader'ını yeniden kuracak.
Veri kaybı beklenmiyor, ancak yanlış disk üzerinde çalıştırılırsa
boot yapılandırması bozulabilir.

EOF
    confirm "Devam edilsin mi?"

    preflight
    mount_filesystems
    repair_grub

    log "Tüm adımlar başarılı."
    log "Şimdi reboot edebilirsiniz: sudo reboot"
    log "Reboot öncesi Ventoy USB'sini çıkarın, BIOS doğal sırasıyla sda EFI'sini okusun."
}

main "$@"
